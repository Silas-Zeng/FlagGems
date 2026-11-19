# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused pairwise Q/K rotation and BF16 KV cache writes."""
import torch
from flag_gems.runtime import torch_device_fn
import triton
import triton.language as tl


@triton.jit
def _rope_cache_pair(POS, X, CACHE, row, head, dim, stride: tl.constexpr,
           M, ROWS: tl.constexpr):
    valid = row < M
    position = tl.load(POS+row, valid, 0)
    position = tl.where(position < 0, position+ROWS, position)
    x1 = tl.load(X+row*stride+head*128+dim, valid, 0.).to(tl.float32)
    x2 = tl.load(X+row*stride+head*128+dim+64, valid, 0.).to(tl.float32)
    cos = tl.load(CACHE+position*128+dim, valid, 0.).to(tl.float32)
    sin = tl.load(CACHE+position*128+dim+64, valid, 0.).to(tl.float32)
    return x1*cos-x2*sin, x2*cos+x1*sin


@triton.jit(do_not_specialize=["M", "ACTUAL"])
def _rope_cache_kernel(POS, Q, K, V, CACHE, O, KC, VC, SLOTS, M, ACTUAL,
              QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
              ROWS: tl.constexpr, BS: tl.constexpr,
              KC0: tl.constexpr, KC1: tl.constexpr, KC2: tl.constexpr,
              VC0: tl.constexpr, VC1: tl.constexpr, VC2: tl.constexpr,
              FUSE_KV: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    qgrid = tl.cdiv(M*1024, BLOCK)
    if pid < qgrid:
        i = pid*BLOCK+tl.arange(0, BLOCK)
        row, head, dim = i//1024, (i//64)%16, i%64
        first, second = _rope_cache_pair(POS, Q, CACHE, row, head, dim, QS, M, ROWS)
        out = row*2304+head*128+dim
        tl.store(O+out, first, row < M)
        tl.store(O+out+64, second, row < M)
    else:
        i = (pid-qgrid)*BLOCK+tl.arange(0, BLOCK)
        row, head, dim = i//128, (i//64)%2, i%64
        first, second = _rope_cache_pair(POS, K, CACHE, row, head, dim, KS, M, ROWS)
        out = row*2304+2048+head*128+dim
        tl.store(O+out, first, row < M)
        tl.store(O+out+64, second, row < M)
        if FUSE_KV:
            slot = tl.load(SLOTS+row, row < ACTUAL, -1)
            valid = (row < M)&(row < ACTUAL)&(slot >= 0)
            safe_slot = tl.maximum(slot, 0)
            ko = (safe_slot//BS)*KC0+(safe_slot%BS)*KC1+head*KC2+dim
            vo = (safe_slot//BS)*VC0+(safe_slot%BS)*VC1+head*VC2+dim
            v1 = tl.load(V+row*VS+head*128+dim, valid, 0.)
            v2 = tl.load(V+row*VS+head*128+dim+64, valid, 0.)
            tl.store(KC+ko, first, valid)
            tl.store(KC+ko+64, second, valid)
            tl.store(VC+vo, v1, valid)
            tl.store(VC+vo+64, v2, valid)


def rope_qk_cache(positions, query, key, value, cache, key_cache, value_cache, slots):
    m = query.shape[0]
    if (query.ndim != 2 or query.shape[1] != 2048 or key.shape != (m, 256)
            or value.shape != (m, 256) or query.dtype != torch.bfloat16
            or any(t.dtype != query.dtype for t in (key, value, cache, key_cache, value_cache))
            or query.device.type != "cuda"
            or any(t.device != query.device for t in (positions, key, value, cache, key_cache, value_cache, slots))
            or positions.shape != (m,) or positions.dtype not in (torch.int32, torch.int64)
            or slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64)
            or slots.numel() > m or not slots.is_contiguous() or not positions.is_contiguous()
            or cache.ndim != 2 or cache.shape[1] != 128 or not cache.is_contiguous()
            or key_cache.ndim != 4 or key_cache.shape != value_cache.shape
            or key_cache.shape[2:] != (2, 128) or key_cache.stride(-1) != 1
            or value_cache.stride(-1) != 1
            or any(t.stride(-1) != 1 for t in (query, key, value))):
        raise ValueError("unsupported fused RoPE/cache operands")
    output = torch.empty((m, 2304), device=query.device, dtype=query.dtype)
    if m:
        with torch_device_fn.device(query.device):
            _rope_cache_kernel[(triton.cdiv(m*1024, 1024)+triton.cdiv(m*128, 1024),)](
                positions, query, key, value, cache, output, key_cache, value_cache, slots,
                m, slots.numel(), query.stride(0), key.stride(0), value.stride(0),
                cache.shape[0], key_cache.shape[1], *key_cache.stride()[:3],
                *value_cache.stride()[:3], True, BLOCK=1024, num_warps=4,
            )
    return output
