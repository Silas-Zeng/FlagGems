# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless BF16 KV gathering for C500 single-request long prefill."""

import functools

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@functools.lru_cache(maxsize=None)
def _is_c500(device_index):
    return "C500" in torch.cuda.get_device_name(device_index)


@triton.jit(do_not_specialize=["N"])
def _gather_prefill_kv(K, V, TABLE, CU_K, OUT, N,
                       EXACT_LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, head, dim = i // 256, (i % 256) // 128, i % 128
    if EXACT_LENGTH:
        length = N
    else:
        length = tl.load(CU_K + 1) - tl.load(CU_K)
    valid = (i < N * 256) & (token < length)
    page = tl.load(TABLE + token // 16, valid, 0).to(tl.int32)
    source = page * 8192 + (token % 16) * 128 + head * 4096 + dim
    key = tl.load(K + source, valid, 0)
    value = tl.load(V + source, valid, 0)
    tl.store(OUT + i, key, i < N * 256)
    tl.store(OUT + N * 256 + i, value, i < N * 256)


def gather_bf16_prefill_cache(key, value, block_table, cu_seqlens_k, max_seqlen_k,
                              *, exact_length=False):
    """Return dense K/V views, or None when the optimized layout is unsupported.

    The cumulative lengths describe one request starting at zero. Valid cache
    values are copied as BF16, with no conversion or change to the paged cache.
    The GPU length limits gathers when max_seqlen_k is an upper bound.
    A caller whose CPU metadata supplies the exact length can omit that read.
    """
    if (key.dtype != torch.bfloat16 or key.device.type != "cuda"
            or key.ndim != 4 or key.shape != value.shape
            or key.shape[1:] != (16, 2, 128)
            or key.stride() != (8192, 128, 4096, 1)
            or value.stride() != key.stride()
            or value.dtype != key.dtype or value.device != key.device
            or key.shape[0] > (2**31 - 1) // 8192
            or block_table is None or block_table.ndim != 2
            or block_table.shape[0] != 1 or block_table.stride(1) != 1
            or block_table.dtype != torch.int32 or block_table.device != key.device
            or not 1024 <= max_seqlen_k <= 4096
            or max_seqlen_k > block_table.shape[1] * 16
            or cu_seqlens_k is None or cu_seqlens_k.shape != (2,)
            or cu_seqlens_k.dtype != torch.int32 or cu_seqlens_k.device != key.device
            or not cu_seqlens_k.is_contiguous() or not _is_c500(key.device.index)):
        return None
    output = torch.empty((2, max_seqlen_k, 2, 128), device=key.device, dtype=key.dtype)
    with torch_device_fn.device(key.device):
        _gather_prefill_kv[(triton.cdiv(max_seqlen_k * 256, 1024),)](
            key, value, block_table, cu_seqlens_k, output, max_seqlen_k,
            exact_length, 1024,
            num_warps=4,
        )
    return output.unbind(0)
