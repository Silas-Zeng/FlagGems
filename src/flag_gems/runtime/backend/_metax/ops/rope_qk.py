# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pairwise Q/K RoPE with one shared output allocation."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit(do_not_specialize=["M"])
def _qk_rope_kernel(POS, Q, K, CACHE, O, M, QS: tl.constexpr, KS: tl.constexpr,
                    ROWS: tl.constexpr, SHARED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    row, part = i//1152, i%1152
    head, dim = part//64, part%64
    valid = row < M
    if SHARED:
        source = Q+row*2560+head*128
    else:
        source = tl.where(head < 16, Q+row*QS+head*128, K+row*KS+(head-16)*128)
    position = tl.load(POS+row, valid, 0)
    position = tl.where(position < 0, position+ROWS, position)
    x1 = tl.load(source+dim, valid, 0.).to(tl.float32)
    x2 = tl.load(source+dim+64, valid, 0.).to(tl.float32)
    cos = tl.load(CACHE+position*128+dim, valid, 0.).to(tl.float32)
    sin = tl.load(CACHE+position*128+dim+64, valid, 0.).to(tl.float32)
    out = row*2304+head*128+dim
    tl.store(O+out, x1*cos-x2*sin, valid)
    tl.store(O+out+64, x2*cos+x1*sin, valid)


def rope_qk(positions, query, key, cache):
    if (query.ndim != 2 or key.ndim != 2 or query.shape[1] != 2048
            or key.shape != (query.shape[0], 256)
            or positions.ndim != 1 or positions.numel() != query.shape[0]
            or positions.dtype not in (torch.int32, torch.int64)
            or query.dtype != torch.bfloat16 or key.dtype != query.dtype
            or cache.dtype != query.dtype or cache.ndim != 2 or cache.shape[1] != 128
            or query.device.type != "cuda" or query.device != key.device
            or query.device != positions.device or query.device != cache.device
            or query.stride(1) != 1 or key.stride(1) != 1
            or not positions.is_contiguous() or not cache.is_contiguous()):
        raise ValueError("unsupported fused Q/K RoPE operands")
    m = query.shape[0]
    output = torch.empty((m, 2304), dtype=query.dtype, device=query.device)
    shared = (query.stride(0) == key.stride(0) == 2560
              and key.data_ptr() == query.data_ptr()+4096)
    if m:
        block, warps = (256, 4) if m <= 8 else ((512, 8) if m <= 64 else (1024, 4))
        with torch_device_fn.device(query.device):
            _qk_rope_kernel[(triton.cdiv(m*1152, block),)](
                positions, query, key, cache, output, m,
                query.stride(0), key.stride(0), cache.shape[0], shared,
                BLOCK=block, num_warps=warps,
            )
    return output
