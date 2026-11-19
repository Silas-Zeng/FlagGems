# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 split-K QKV projection with fused rounding, RoPE and cache update."""

import torch
from flag_gems.runtime import torch_device_fn
from .prefill_linear import _projection_odd_partial

import triton
import triton.language as tl

@triton.jit
def _partial_value(P, index, M, SPLIT:tl.constexpr, valid):
    value=tl.full(index.shape,0.,tl.float32)
    for s in range(SPLIT):
        value+=tl.load(P+s*M*2560+index,valid,0.)
    return value.to(tl.bfloat16).to(tl.float32)

@triton.jit(do_not_specialize=['M','ACTUAL'])
def _reduce_rope_cache(P,POS,CACHE,O,VO,KC,VC,SLOTS,M,ACTUAL,
                      SPLIT:tl.constexpr,ROWS:tl.constexpr,BS:tl.constexpr,
                      KC0:tl.constexpr,KC1:tl.constexpr,KC2:tl.constexpr,
                      VC0:tl.constexpr,VC1:tl.constexpr,VC2:tl.constexpr,
                      FUSE_KV:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    qgrid=tl.cdiv(M*1024,BLOCK)
    if pid<qgrid:
        i=pid*BLOCK+tl.arange(0,BLOCK)
        row,head,dim=i//1024,(i//64)%16,i%64
        index=row*2560+head*128+dim
        valid=row<M
        x1=_partial_value(P,index,M,SPLIT,valid)
        x2=_partial_value(P,index+64,M,SPLIT,valid)
        pos=tl.load(POS+row,valid,0)
        pos=tl.where(pos<0,pos+ROWS,pos)
        c=tl.load(CACHE+pos*128+dim,valid,0.).to(tl.float32)
        s=tl.load(CACHE+pos*128+dim+64,valid,0.).to(tl.float32)
        out=row*2304+head*128+dim
        tl.store(O+out,x1*c-x2*s,valid)
        tl.store(O+out+64,x2*c+x1*s,valid)
    else:
        i=(pid-qgrid)*BLOCK+tl.arange(0,BLOCK)
        row,head,dim=i//128,(i//64)%2,i%64
        index=row*2560+2048+head*128+dim
        valid=row<M
        x1=_partial_value(P,index,M,SPLIT,valid)
        x2=_partial_value(P,index+64,M,SPLIT,valid)
        v1=_partial_value(P,index+256,M,SPLIT,valid)
        v2=_partial_value(P,index+320,M,SPLIT,valid)
        pos=tl.load(POS+row,valid,0)
        pos=tl.where(pos<0,pos+ROWS,pos)
        c=tl.load(CACHE+pos*128+dim,valid,0.).to(tl.float32)
        s=tl.load(CACHE+pos*128+dim+64,valid,0.).to(tl.float32)
        k1=x1*c-x2*s;k2=x2*c+x1*s
        out=row*2304+2048+head*128+dim
        tl.store(O+out,k1,valid);tl.store(O+out+64,k2,valid)
        tl.store(VO+row*2560+head*128+dim,v1,valid);tl.store(VO+row*2560+head*128+dim+64,v2,valid)
        if FUSE_KV:
            slot=tl.load(SLOTS+row,row<ACTUAL,-1)
            good=valid&(row<ACTUAL)&(slot>=0)
            slot=tl.maximum(slot,0)
            ko=(slot//BS)*KC0+(slot%BS)*KC1+head*KC2+dim
            vo=(slot//BS)*VC0+(slot%BS)*VC1+head*VC2+dim
            tl.store(KC+ko,k1,good);tl.store(KC+ko+64,k2,good)
            tl.store(VC+vo,v1,good);tl.store(VC+vo+64,v2,good)


def qkv_rope_cache(a, packed, positions, cache, key_cache, value_cache, slots):
    """Specialized C500 decode path; return Q/K and the strided raw V view."""
    m = a.shape[0]
    if (a.shape != (m, 2048) or not 33 <= m <= 64
            or a.dtype != torch.bfloat16 or a.device.type != "cuda"
            or not a.is_contiguous() or packed.shape != (20, 32, 128, 64)
            or packed.dtype != a.dtype or not packed.is_contiguous()
            or positions.shape != (m,) or not positions.is_contiguous()
            or positions.dtype not in (torch.int32, torch.int64)
            or cache.ndim != 2 or cache.shape[1] != 128 or not cache.is_contiguous()
            or key_cache.ndim != 4 or key_cache.shape != value_cache.shape
            or key_cache.shape[2:] != (2, 128)
            or key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1
            or slots.ndim != 1 or slots.numel() > m or not slots.is_contiguous()
            or slots.dtype not in (torch.int32, torch.int64)
            or any(t.device != a.device for t in (packed, positions, cache, key_cache, value_cache, slots))
            or any(t.dtype != a.dtype for t in (cache, key_cache, value_cache))):
        raise ValueError("unsupported fused QKV/RoPE/cache operands")
    partials = torch.empty((2, m, 2560), device=a.device, dtype=torch.float32)
    output = torch.empty((m, 2304), device=a.device, dtype=a.dtype)
    # Match the value view's strides and storage offset on the unfused path.
    value_storage = torch.empty((m, 2560), device=a.device, dtype=a.dtype)
    value = value_storage[:, 2304:]
    with torch_device_fn.device(a.device):
        _projection_odd_partial[(40, 2)](
            a, packed, partials, m, 2560, 2048, 64, 64, 2,
            num_warps=8, num_stages=4,
        )
        _reduce_rope_cache[(m + triton.cdiv(m * 128, 1024),)](
            partials, positions, cache, output, value, key_cache, value_cache,
            slots, m, slots.numel(), 2, cache.shape[0], key_cache.shape[1],
            *key_cache.stride()[:3], *value_cache.stride()[:3], True, 1024,
            num_warps=4,
        )
    return output, value
