# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 add/RMSNorm retaining a complete row in registers."""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit(do_not_specialize=["M"])
def _residual_rms_kernel(X, R, W, O, RO, M, EPS: tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,2048)
    x=tl.load(X+row*2048+col,row<M,0.).to(tl.float32)
    residual=tl.load(R+row*2048+col,row<M,0.).to(tl.float32)
    total=x+residual
    variance=tl.sum(total*total,0)/2048.
    normalized=total*tl.rsqrt(variance+EPS)
    weight=tl.load(W+col).to(tl.float32)
    tl.store(O+row*2048+col,normalized*weight,row<M)
    tl.store(RO+row*2048+col,total,row<M)


def residual_rms(x, residual, weight, epsilon):
    if (x.ndim != 2 or x.shape[1] != 2048 or residual.shape != x.shape
            or weight.shape != (2048,) or x.dtype != torch.bfloat16
            or residual.dtype != x.dtype or weight.dtype != x.dtype
            or x.device.type != "cuda" or residual.device != x.device
            or weight.device != x.device or not x.is_contiguous()
            or not residual.is_contiguous() or not weight.is_contiguous()):
        raise ValueError("unsupported residual RMSNorm operands")
    output,residual_out=torch.empty_like(x),torch.empty_like(residual)
    m=x.shape[0]
    if m:
        with torch_device_fn.device(x.device):
            _residual_rms_kernel[(m,)](x,residual,weight,output,residual_out,m,
                                      epsilon,num_warps=8 if m<=64 else 4)
    return output,residual_out
