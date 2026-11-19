# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton paged BF16 GQA decode for the C500 interleaved HND cache.

Each program shares one KV head across its eight query heads. Head dimensions
are tiled in 32 or 64 values to reduce operand-conversion overhead. A second
Triton kernel merges FP32 split outputs using their log-sum-exp weights.
No external compiler, native extension or host read of sequence lengths is
used. Sequence lengths and page tables remain runtime GPU inputs.
"""

import functools
import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("VLLM_FL_METAX_TRITON_DECODE", "1") == "1"
_SPLITS = int(os.environ.get("FLAGGEMS_METAX_DECODE_SPLITS", "0"))
_BLOCK_N = int(os.environ.get("FLAGGEMS_METAX_DECODE_BLOCK_N", "0"))


@triton.jit(do_not_specialize=["BATCH"])
def _paged_gqa_split(
    Q, K, V, TABLE, LENGTHS, PARTIAL, LSE, BATCH,
    Q_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Adjacent programs process the two KV heads of the same request.
    head = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    h = tl.arange(0, 16)
    d = tl.arange(0, 64)
    n = tl.arange(0, BLOCK_N)
    q0 = tl.load(
        Q + batch * Q_STRIDE + (head * 8 + h[:, None]) * 128
        + d[None, :], h[:, None] < 8, 0.0,
    )
    q1 = tl.load(
        Q + batch * Q_STRIDE + (head * 8 + h[:, None]) * 128
        + d[None, :] + 64, h[:, None] < 8, 0.0,
    )
    length = tl.load(LENGTHS + batch)
    width = tl.cdiv(length, SPLITS * BLOCK_N) * BLOCK_N
    begin = split * width
    end = tl.minimum(begin + width, length)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, 64), tl.float32)
    acc1 = tl.zeros((16, 64), tl.float32)
    for start in range(begin, end, BLOCK_N):
        tokens = start + n
        valid = tokens < end
        pages = tl.load(TABLE + batch * TABLE_STRIDE + tokens // 16, valid, 0)
        offsets = pages * 8192 + head * 4096 + (tokens % 16) * 128
        # Keep each QK/PV operand tile narrow instead of converting D=128.
        k0 = tl.load(K + offsets[None, :] + d[:, None],
                     valid[None, :], 0.0)
        scores = tl.dot(q0, k0, out_dtype=tl.float32)
        k1 = tl.load(K + offsets[None, :] + d[:, None] + 64,
                     valid[None, :], 0.0)
        scores = tl.dot(q1, k1, scores)
        scores = tl.where(valid[None, :], scores * 0.1275174308245987,
                          -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_maximum)
        probabilities = tl.exp2(scores - new_maximum[:, None])
        probabilities_bf16 = probabilities.to(tl.bfloat16)
        acc0 *= alpha[:, None]
        v0 = tl.load(V + offsets[:, None] + d[None, :],
                     valid[:, None], 0.0)
        acc0 = tl.dot(probabilities_bf16, v0, acc0)
        acc1 *= alpha[:, None]
        v1 = tl.load(V + offsets[:, None] + d[None, :] + 64,
                     valid[:, None], 0.0)
        acc1 = tl.dot(probabilities_bf16, v1, acc1)
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        maximum = new_maximum
    base = ((split * BATCH + batch) * 16 + head * 8 + h)
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    tl.store(PARTIAL + base[:, None] * 128 + d[None, :],
             acc0 / safe_denominator[:, None], h[:, None] < 8)
    tl.store(PARTIAL + base[:, None] * 128 + d[None, :] + 64,
             acc1 / safe_denominator[:, None], h[:, None] < 8)
    logsum = tl.where(end > begin, maximum + tl.log2(denominator),
                      -float("inf"))
    tl.store(LSE + base, logsum, h < 8)


@triton.jit(do_not_specialize=["BATCH"])
def _paged_gqa_split_narrow(
    Q, K, V, TABLE, LENGTHS, PARTIAL, LSE, BATCH,
    Q_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Token-major QK scores feed dimension-major PV accumulators directly.
    # This keeps probabilities in the same orientation between dot products.
    # Adjacent programs process the two KV heads of the same request.
    head = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    h = tl.arange(0, 16)
    d = tl.arange(0, 32)
    n = tl.arange(0, BLOCK_N)
    q0 = tl.load(
        Q + batch * Q_STRIDE + (head * 8 + h[:, None]) * 128
        + d[None, :], h[:, None] < 8, 0.0,
    )
    q1 = tl.load(
        Q + batch * Q_STRIDE + (head * 8 + h[:, None]) * 128
        + d[None, :] + 32, h[:, None] < 8, 0.0,
    )
    q2 = tl.load(
        Q + batch * Q_STRIDE + (head * 8 + h[:, None]) * 128
        + d[None, :] + 64, h[:, None] < 8, 0.0,
    )
    q3 = tl.load(
        Q + batch * Q_STRIDE + (head * 8 + h[:, None]) * 128
        + d[None, :] + 96, h[:, None] < 8, 0.0,
    )
    length = tl.load(LENGTHS + batch)
    width = tl.cdiv(length, SPLITS * BLOCK_N) * BLOCK_N
    begin = split * width
    end = tl.minimum(begin + width, length)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((32, 16), tl.float32)
    acc1 = tl.zeros((32, 16), tl.float32)
    acc2 = tl.zeros((32, 16), tl.float32)
    acc3 = tl.zeros((32, 16), tl.float32)
    for start in range(begin, end, BLOCK_N):
        tokens = start + n
        valid = tokens < end
        pages = tl.load(TABLE + batch * TABLE_STRIDE + tokens // 16, valid, 0)
        offsets = pages * 8192 + head * 4096 + (tokens % 16) * 128
        # Keep each QK/PV operand tile narrow instead of converting D=128.
        k0 = tl.load(K + offsets[None, :] + d[:, None],
                     valid[None, :], 0.0)
        scores = tl.dot(tl.trans(k0), tl.trans(q0), out_dtype=tl.float32)
        k1 = tl.load(K + offsets[None, :] + d[:, None] + 32,
                     valid[None, :], 0.0)
        scores = tl.dot(tl.trans(k1), tl.trans(q1), scores)
        k2 = tl.load(K + offsets[None, :] + d[:, None] + 64,
                     valid[None, :], 0.0)
        scores = tl.dot(tl.trans(k2), tl.trans(q2), scores)
        k3 = tl.load(K + offsets[None, :] + d[:, None] + 96,
                     valid[None, :], 0.0)
        scores = tl.dot(tl.trans(k3), tl.trans(q3), scores)
        scores = tl.where(valid[:, None], scores * 0.1275174308245987,
                          -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp2(maximum - new_maximum)
        probabilities = tl.exp2(scores - new_maximum[None, :])
        probabilities_bf16 = probabilities.to(tl.bfloat16)
        acc0 *= alpha[None, :]
        v0 = tl.load(V + offsets[None, :] + d[:, None],
                     valid[None, :], 0.0)
        acc0 = tl.dot(v0, probabilities_bf16, acc0)
        acc1 *= alpha[None, :]
        v1 = tl.load(V + offsets[None, :] + d[:, None] + 32,
                     valid[None, :], 0.0)
        acc1 = tl.dot(v1, probabilities_bf16, acc1)
        acc2 *= alpha[None, :]
        v2 = tl.load(V + offsets[None, :] + d[:, None] + 64,
                     valid[None, :], 0.0)
        acc2 = tl.dot(v2, probabilities_bf16, acc2)
        acc3 *= alpha[None, :]
        v3 = tl.load(V + offsets[None, :] + d[:, None] + 96,
                     valid[None, :], 0.0)
        acc3 = tl.dot(v3, probabilities_bf16, acc3)
        denominator = denominator * alpha + tl.sum(probabilities, 0)
        maximum = new_maximum
    base = ((split * BATCH + batch) * 16 + head * 8 + h)
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    tl.store(PARTIAL + base[:, None] * 128 + d[None, :],
             tl.trans(acc0) / safe_denominator[:, None], h[:, None] < 8)
    tl.store(PARTIAL + base[:, None] * 128 + d[None, :] + 32,
             tl.trans(acc1) / safe_denominator[:, None], h[:, None] < 8)
    tl.store(PARTIAL + base[:, None] * 128 + d[None, :] + 64,
             tl.trans(acc2) / safe_denominator[:, None], h[:, None] < 8)
    tl.store(PARTIAL + base[:, None] * 128 + d[None, :] + 96,
             tl.trans(acc3) / safe_denominator[:, None], h[:, None] < 8)
    logsum = tl.where(end > begin, maximum + tl.log2(denominator),
                      -float("inf"))
    tl.store(LSE + base, logsum, h < 8)


@triton.jit(do_not_specialize=["BATCH"])
def _paged_gqa_merge(PARTIAL, LSE, OUT, BATCH,
                     SPLITS: tl.constexpr, BLOCK_S: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, 128)
    lse = tl.load(LSE + s * BATCH * 16 + row, s < SPLITS, -float("inf"))
    maximum = tl.max(lse, 0)
    safe_maximum = tl.where(maximum > -float("inf"), maximum, 0.0)
    weights = tl.exp2(lse - safe_maximum)
    denominator = tl.sum(weights, 0)
    values = tl.load(PARTIAL + (s[:, None] * BATCH * 16 + row) * 128
                     + d[None, :], s[:, None] < SPLITS, 0.0)
    output = tl.sum(values * weights[:, None], 0)
    output /= tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + row * 128 + d, output)


@functools.lru_cache(maxsize=None)
def _is_c500(device):
    return "C500" in torch.cuda.get_device_name(device)


def try_bf16_decode(q, key_cache, value_cache, output, block_table, lengths, scale):
    """Write the supported single-token decode into output; otherwise decline."""
    if not _ENABLED or q.ndim != 4:
        return False
    batch = q.shape[0]
    if (q.shape[1:] != (1, 16, 128) or not 1 <= batch <= 64
            or q.dtype != torch.bfloat16 or q.device.type != "cuda"
            or q.stride(-1) != 1 or q.stride(2) != 128 or q.stride(0) < 2048
            or output.shape != q.shape or not output.is_contiguous()
            or key_cache.ndim != 4 or key_cache.shape != value_cache.shape
            or key_cache.shape[1:] != (16, 2, 128)
            or key_cache.stride() != (8192, 128, 4096, 1)
            or value_cache.stride() != key_cache.stride()
            or any(t.dtype != q.dtype or t.device != q.device
                   for t in (key_cache, value_cache, output))
            or block_table is None or block_table.ndim != 2
            or block_table.shape[0] < batch or block_table.shape[1] == 0
            or block_table.stride(1) != 1 or block_table.stride(0) < block_table.shape[1]
            or block_table.dtype != torch.int32 or block_table.device != q.device
            or lengths is None or lengths.shape != (batch,)
            or lengths.dtype != torch.int32 or lengths.device != q.device
            or not lengths.is_contiguous() or abs(scale - 128 ** -0.5) > 1e-8
            or not _is_c500(q.device.index)):
        return False
    if _SPLITS < 0 or _BLOCK_N not in (0, 32, 64, 128):
        raise ValueError("invalid C500 Triton decode configuration")
    # Enough independent programs for short batches; no sequence-length host read.
    splits = _SPLITS or (64 if batch == 1 else 32 if batch == 2
                         else 24 if batch <= 4 else 12 if batch <= 8 else 6)
    narrow = batch >= 56
    block_n = _BLOCK_N or (32 if narrow else 64)
    partials = torch.empty((splits, batch, 16, 128), device=q.device, dtype=torch.float32)
    lse = torch.empty((splits, batch, 16), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        kernel = _paged_gqa_split_narrow if narrow else _paged_gqa_split
        kernel[(2, batch, splits)](
            q, key_cache, value_cache, block_table, lengths, partials, lse, batch,
            q.stride(0), block_table.stride(0), splits, block_n,
            num_warps=1 if narrow else 4, num_stages=1 if narrow else 2,
            scenario="mla" if narrow else "mla;singleshm",
            pipeline_load_num=-1 if narrow else 2,
        )
        _paged_gqa_merge[(batch * 16,)](
            partials, lse, output, batch, splits, triton.next_power_of_2(splits),
            num_warps=1 if splits <= 16 else 4,
        )
    return True
