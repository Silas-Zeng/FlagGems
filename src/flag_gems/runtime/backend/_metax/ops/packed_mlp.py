# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""BF16 MLP kernels with contiguous weight tiles and a rounded SwiGLU epilogue."""

import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .mm import mm_kernel_splitk_reduce, mm_out

PACK_N = 128
PACK_K = 64
_PREFILL_SMEM_OPT = os.environ.get("FLAGGEMS_METAX_PREFILL_SMEM_OPT", "1") == "1"
_BSM_PREFILL = os.environ.get("FLAGGEMS_METAX_BSM_PREFILL", "1") == "1"
_BSM_LINEAR_PREFILL = os.environ.get("FLAGGEMS_METAX_BSM_LINEAR_PREFILL", "1") == "1"


@triton.jit
def _pack_kernel(W, P, N: tl.constexpr, K: tl.constexpr,
                 INTERLEAVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, k = i // K, i % K
    if INTERLEAVE:
        source_n = n // 2 + (n % 2) * (N // 2)
    else:
        source_n = n
    dest = ((n // 128) * (K // 64) + k // 64) * (128 * 64)
    dest += (n % 128) * 64 + k % 64
    value = tl.load(W + source_n * K + k, i < N * K, 0.0)
    tl.store(P + dest, value, i < N * K)


def pack_mlp_weight(weight, *, interleave=False, out=None):
    if (weight.ndim != 2 or weight.dtype != torch.bfloat16
            or not weight.is_contiguous() or weight.device.type != "cuda"):
        raise ValueError("packing requires a contiguous CUDA BF16 matrix")
    n, k = weight.shape
    if n % PACK_N or k % PACK_K:
        raise ValueError("weight dimensions must align to the packing tile")
    shape = (n // PACK_N, k // PACK_K, PACK_N, PACK_K)
    if out is None:
        out = torch.empty(shape, dtype=weight.dtype, device=weight.device)
    elif (out.shape != shape or out.dtype != weight.dtype
          or out.device != weight.device or not out.is_contiguous()):
        raise ValueError("packed output has the wrong shape, dtype, device or layout")
    with torch_device_fn.device(weight.device):
        _pack_kernel[(triton.cdiv(n * k, 1024),)](
            weight, out, n, k, INTERLEAVE=interleave, BLOCK=1024, num_warps=4
        )
    return out


def _check_operands(a, weight, packed, n, k):
    if (
        a.ndim != 2 or a.shape[1] != k or weight.shape != (n, k)
        or a.dtype != torch.bfloat16 or weight.dtype != a.dtype
        or packed.dtype != a.dtype or a.device != weight.device
        or a.device != packed.device or a.device.type != "cuda"
        or packed.shape != (n // 128, k // 64, 128, 64)
        or not packed.is_contiguous() or not weight.is_contiguous()
    ):
        raise ValueError("unsupported packed MLP operands")


@triton.jit(do_not_specialize=["M"])
def _gate_up_kernel(A, P, O, M, BM: tl.constexpr, BN: tl.constexpr,
                    PREFILL: tl.constexpr = False, BK: tl.constexpr = 64):
    # Packed columns alternate gate/up. One dot computes both, retaining the
    # BF16 GEMM-output rounding before the FP32 compiled SiLU*up expression.
    if PREFILL:
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = 6144 // BN
        group = pid // (16 * grid_n)
        first_m = group * 16
        group_m = tl.minimum(grid_m - first_m, 16)
        pid_m = first_m + (pid % (16 * grid_n)) % group_m
        pid_n = (pid % (16 * grid_n)) // group_m
    else:
        pid_m = 0
        pid_n = tl.program_id(0)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * (2 * BN) + tl.arange(0, 2 * BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 2 * BN), tl.float32)
    for j in range(2048 // BK):
        k = j * BK + rk
        a = tl.load(A + rm[:, None] * 2048 + k[None, :], rm[:, None] < M, 0.0)
        index = ((rn[:, None] // 128) * 32 + k[None, :] // 64) * (128 * 64)
        index += (rn[:, None] % 128) * 64 + k[None, :] % 64
        w = tl.load(P + index)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32)
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(tl.reshape(rounded, (BM, BN, 2)))
    value = (gate * tl.sigmoid(gate)) * up
    col = pid_n * BN + tl.arange(0, BN)
    tl.store(O + rm[:, None] * 6144 + col[None, :], value, rm[:, None] < M)


@triton.jit
def _activation_kernel(X, Y, NUMEL, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // 6144, i % 6144
    gate = tl.load(X + row * 12288 + col, i < NUMEL, 0.0).to(tl.float32)
    up = tl.load(X + row * 12288 + 6144 + col, i < NUMEL, 0.0).to(tl.float32)
    tl.store(Y + i, (gate * tl.sigmoid(gate)) * up, i < NUMEL)


def packed_gate_up_silu(a, weight, packed, *, large_prefill=False):
    _check_operands(a, weight, packed, 12288, 2048)
    m = a.shape[0]
    output = torch.empty((m, 6144), dtype=a.dtype, device=a.device)
    if (_BSM_PREFILL and large_prefill and 1792 <= m <= 2048
            and a.is_contiguous()):
        from .bsm_gate import packed_gate_into

        if packed_gate_into(a, packed, output):
            return output
    if 0 < m <= 64 and a.is_contiguous():
        bm, bn, warps = (32, 64, 8) if m <= 32 else (64, 64, 16)
        with torch_device_fn.device(a.device):
            _gate_up_kernel[(6144 // bn,)](
                a, packed, output, m, BM=bm, BN=bn, num_warps=warps, num_stages=2
            )
    elif large_prefill and 1024 <= m <= 2048 and a.is_contiguous():
        with torch_device_fn.device(a.device):
            _gate_up_kernel[(triton.cdiv(m, 256) * 48,)](
                a, packed, output, m, BM=256, BN=128, BK=32, PREFILL=True,
                num_warps=8, num_stages=2,
                scenario="smemIndexOpt" if _PREFILL_SMEM_OPT else "",
            )
    elif 128 <= m <= 2048 and a.is_contiguous():
        with torch_device_fn.device(a.device):
            _gate_up_kernel[(triton.cdiv(m, 128) * 96,)](
                a, packed, output, m, BM=128, BN=64, PREFILL=True,
                num_warps=8, num_stages=2,
            )
    else:
        intermediate = torch.empty((m, 12288), dtype=a.dtype, device=a.device)
        mm_out(a, weight.t(), out=intermediate)
        if m:
            with torch_device_fn.device(a.device):
                _activation_kernel[(triton.cdiv(m * 6144, 1024),)](
                    intermediate, output, m * 6144, BLOCK=1024, num_warps=4
                )
    return output


@triton.jit(do_not_specialize=["M"])
def _down_partial_kernel(A, P, O, M, SPLIT: tl.constexpr = 4):
    rm = tl.arange(0, 64)
    rn = tl.program_id(0) * 128 + tl.arange(0, 128)
    split = tl.program_id(1)
    rk = tl.arange(0, 64)
    acc = tl.zeros((64, 128), tl.float32)
    for j in range(96 // SPLIT):
        kb = split * (96 // SPLIT) + j
        k = kb * 64 + rk
        a = tl.load(A + rm[:, None] * 6144 + k[None, :], rm[:, None] < M, 0.0)
        index = ((rn[:, None] // 128) * 96 + kb) * (128 * 64)
        index += (rn[:, None] % 128) * 64 + rk[None, :]
        w = tl.load(P + index)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32)
    tl.store(O + (split * M + rm[:, None]) * 2048 + rn[None, :], acc, rm[:, None] < M)


@triton.jit(do_not_specialize=["M"])
def _down_prefill_three_kernel(A, W, O, M, N: tl.constexpr, K: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PACKED: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    pm, pn = pid % nm, pid // nm
    rm = pm*BM + tl.arange(0, BM)
    rn = pn*(3*BN) + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    c0 = tl.zeros((BM, BN), tl.float32)
    c1 = tl.zeros((BM, BN), tl.float32)
    c2 = tl.zeros((BM, BN), tl.float32)
    for j in range(K//BK):
        k = j*BK + rk
        a = tl.load(A + rm[:, None]*K + k[None, :], rm[:, None] < M, 0.)
        if PACKED:
            i0 = ((rn[:, None]//128)*(K//64)+k[None, :]//64)*8192 + (rn[:, None]%128)*64 + k[None, :]%64
            r1, r2 = rn+BN, rn+2*BN
            i1 = ((r1[:, None]//128)*(K//64)+k[None, :]//64)*8192 + (r1[:, None]%128)*64 + k[None, :]%64
            i2 = ((r2[:, None]//128)*(K//64)+k[None, :]//64)*8192 + (r2[:, None]%128)*64 + k[None, :]%64
        else:
            i0 = rn[:, None]*K + k[None, :]
            i1 = (rn[:, None]+BN)*K + k[None, :]
            i2 = (rn[:, None]+2*BN)*K + k[None, :]
        w0 = tl.load(W+i0, rn[:, None] < N, 0.)
        w1 = tl.load(W+i1, rn[:, None]+BN < N, 0.)
        w2 = tl.load(W+i2, rn[:, None]+2*BN < N, 0.)
        c0 = tl.dot(a, tl.trans(w0), c0, out_dtype=tl.float32)
        c1 = tl.dot(a, tl.trans(w1), c1, out_dtype=tl.float32)
        c2 = tl.dot(a, tl.trans(w2), c2, out_dtype=tl.float32)
    tl.store(O+rm[:, None]*N+rn[None, :], c0, (rm[:, None] < M)&(rn[None, :] < N))
    tl.store(O+rm[:, None]*N+rn[None, :]+BN, c1, (rm[:, None] < M)&(rn[None, :]+BN < N))
    tl.store(O+rm[:, None]*N+rn[None, :]+2*BN, c2, (rm[:, None] < M)&(rn[None, :]+2*BN < N))


def packed_down(a, weight, packed, *, large_prefill=False, small_split=4):
    _check_operands(a, weight, packed, 2048, 6144)
    if small_split not in (4, 6):
        raise ValueError("unsupported Down split count")
    m = a.shape[0]
    output = torch.empty((m, 2048), dtype=a.dtype, device=a.device)
    if (_BSM_LINEAR_PREFILL and large_prefill and 1024 <= m <= 2048
            and a.is_contiguous()):
        from .bsm_gate import packed_linear_into

        if packed_linear_into(a, packed, output):
            return output
    if 32 < m <= 64 and a.is_contiguous():
        partials = torch.empty((small_split, m, 2048), dtype=torch.float32, device=a.device)
        with torch_device_fn.device(a.device):
            _down_partial_kernel[(16, small_split)](
                a, packed, partials, m, SPLIT=small_split, num_warps=8, num_stages=3
            )
            mm_kernel_splitk_reduce[(triton.cdiv(m * 2048, 1024),)](
                partials, output, m, 2048, 2048, 1,
                SPLIT_K=small_split, BLOCK_SIZE=1024, num_warps=4,
            )
    elif large_prefill and 1792 <= m <= 2048 and a.is_contiguous():
        with torch_device_fn.device(a.device):
            _down_prefill_three_kernel[(triton.cdiv(m, 256) * 11,)](
                a, packed, output, m, 2048, 6144, 256, 64, 32, True,
                num_warps=8, num_stages=2,
                scenario="smemOffsetCache" if _PREFILL_SMEM_OPT else "",
            )
    else:
        mm_out(a, weight.t(), out=output)
    return output
