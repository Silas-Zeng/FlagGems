# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import triton
import triton.language as tl

_PREFILL_SMEM_OPT = os.environ.get("FLAGGEMS_METAX_PREFILL_SMEM_OPT", "1") == "1"


@triton.jit(do_not_specialize=["M"])
def _prefill_nn_kernel(A, B, O, M, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GM: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GM*nn)
    first = group*GM
    gm = tl.minimum(nm-first,GM)
    pm = first+(pid%(GM*nn))%gm
    pn = (pid%(GM*nn))//gm
    rm = pm*BM+tl.arange(0,BM)
    rn = pn*BN+tl.arange(0,BN)
    rk = tl.arange(0,BK)
    ap = A+rm[:,None]*K+rk[None,:]
    bp = B+rk[:,None]*N+rn[None,:]
    acc = tl.zeros((BM,BN),tl.float32)
    for _ in range(K//BK):
        a = tl.load(ap,rm[:,None]<M,0.)
        b = tl.load(bp,rn[None,:]<N,0.)
        acc = tl.dot(a,b,acc,out_dtype=tl.float32)
        ap += BK
        bp += BK*N
    tl.store(O+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))



@triton.jit(do_not_specialize=['M'])
def _projection_packed_kernel(A,P,O,M,N:tl.constexpr,K:tl.constexpr,PN:tl.constexpr,PK:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for j in range(K//BK):
        k=j*BK+rk
        a=tl.load(A+rm[:,None]*K+k[None,:],rm[:,None]<M,0)
        idx=((rn[:,None]//PN)*(K//PK)+k[None,:]//PK)*(PN*PK)+(rn[:,None]%PN)*PK+k[None,:]%PK
        w=tl.load(P+idx)
        acc=tl.dot(a,tl.trans(w),acc,out_dtype=tl.float32)
    tl.store(O+rm[:,None]*N+rn[None,:],acc.to(O.dtype.element_ty),(rm[:,None]<M)&(rn[None,:]<N))


def prefill_linear_nn(a, weight, transposed, *, odd_split=False):
    from .mm import mm_out
    if transposed.ndim == 4:
        return projection_linear_packed(a, weight, transposed, odd_split=odd_split)
    if (a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1]
            or transposed.shape != (weight.shape[1], weight.shape[0])
            or a.dtype != torch.bfloat16 or weight.dtype != a.dtype
            or transposed.dtype != a.dtype or a.device != weight.device
            or a.device != transposed.device or a.device.type != "cuda"
            or not transposed.is_contiguous()):
        raise ValueError("unsupported prepacked linear operands")
    m, k = a.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if 1024 <= m <= 2048 and k == 2048 and n in (2048, 2560) and a.is_contiguous():
        _prefill_nn_kernel[(triton.cdiv(m, 256)*triton.cdiv(n, 256),)](
            a, transposed, output, m, n, k, 256, 256, 32, 16,
            num_warps=8, num_stages=2,
        )
    elif m:
        mm_out(a, weight.t(), out=output)
    return output


@triton.jit(do_not_specialize=["M"])
def _projection_odd_partial(A, P, O, M, N: tl.constexpr, K: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    rm = tl.arange(0, 64)
    rn = tl.program_id(0)*BN+tl.arange(0, BN)
    split = tl.program_id(1)
    rk = tl.arange(0, BK)
    steps: tl.constexpr = triton.cdiv(K//BK, SPLIT)
    acc = tl.zeros((64, BN), tl.float32)
    for j in range(steps):
        k = (split*steps+j)*BK+rk
        a = tl.load(A+rm[:, None]*K+k[None, :], (rm[:, None] < M)&(k[None, :] < K), 0.)
        index = ((rn[:, None]//128)*(K//64)+k[None, :]//64)*8192
        index += (rn[:, None]%128)*64+k[None, :]%64
        w = tl.load(P+index, (rn[:, None] < N)&(k[None, :] < K), 0.)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32)
    tl.store(O+(split*M+rm[:, None])*N+rn[None, :], acc,
             (rm[:, None] < M)&(rn[None, :] < N))


def projection_linear_packed(a, weight, packed, *, odd_split=False):
    from flag_gems.runtime import torch_device_fn
    from .mm import mm_out
    if (a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1]
            or a.dtype != torch.bfloat16 or weight.dtype != a.dtype
            or packed.dtype != a.dtype or a.device != weight.device
            or a.device != packed.device or a.device.type != "cuda"
            or weight.shape[1] != 2048 or weight.shape[0] not in (2048, 2560)
            or packed.shape != (weight.shape[0]//128, 32, 128, 64)
            or not packed.is_contiguous()):
        raise ValueError("unsupported packed projection operands")
    m, k = a.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if odd_split and a.is_contiguous() and 33 <= m <= 64 and n == 2048:
        from .mm import mm_kernel_splitk_reduce
        partials = torch.empty((3, m, n), device=a.device, dtype=torch.float32)
        with torch_device_fn.device(a.device):
            _projection_odd_partial[(32, 3)](
                a, packed, partials, m, n, k, 64, 64, 3,
                num_warps=8, num_stages=3,
            )
            mm_kernel_splitk_reduce[(triton.cdiv(m*n, 1024),)](
                partials, output, m, n, n, 1, SPLIT_K=3,
                BLOCK_SIZE=1024, num_warps=4,
            )
    elif a.is_contiguous() and (33 <= m <= 64 or 1024 <= m <= 2048):
        bm, bn, bk = (64, 32, 128) if m <= 64 else (256, 256, 32)
        with torch_device_fn.device(a.device):
            _projection_packed_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                a, packed, output, m, n, k, 128, 64, bm, bn, bk,
                num_warps=8, num_stages=2,
                scenario="smemIndexOpt" if _PREFILL_SMEM_OPT and m >= 1024 else "",
            )
    elif m:
        mm_out(a, weight.t(), out=output)
    return output
