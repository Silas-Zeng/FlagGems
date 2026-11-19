# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fuse the split-K epilogue with the following residual RMSNorm."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .packed_mlp import _check_operands, _down_partial_kernel, packed_down
from .prefill_linear import _projection_odd_partial, prefill_linear_nn
from .residual_rms import residual_rms


@triton.jit(do_not_specialize=["M"])
def _reduce_residual_rms(P, R, W, O, RO, M, SPLIT: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 2048)
    index = row * 2048 + col
    value = tl.full((2048,), 0.0, tl.float32)
    for split in range(SPLIT):
        value += tl.load(P + split * M * 2048 + index)
    # Keep the original BF16 projection result before residual addition.
    rounded = value.to(tl.bfloat16).to(tl.float32)
    total = rounded + tl.load(R + index).to(tl.float32)
    variance = tl.sum(total * total, 0) / 2048.0
    normalized = total * tl.rsqrt(variance + EPS)
    weight = tl.load(W + col).to(tl.float32)
    tl.store(O + index, normalized * weight)
    tl.store(RO + index, total)


def linear_residual_rms(a, weight, packed, residual, norm_weight, epsilon,
                        *, down, split_k, large_prefill=True):
    k = 6144 if down else 2048
    _check_operands(a, weight, packed, 2048, k)
    m = a.shape[0]
    if (residual.shape != (m, 2048) or norm_weight.shape != (2048,)
            or residual.dtype != torch.bfloat16 or norm_weight.dtype != residual.dtype
            or residual.device != a.device or norm_weight.device != a.device
            or not residual.is_contiguous() or not norm_weight.is_contiguous()
            or (down and split_k not in (4, 6))
            or (not down and split_k not in (0, 3))):
        raise ValueError("unsupported fused linear/residual RMSNorm operands")
    if not (33 <= m <= 64 and a.is_contiguous() and split_k):
        if down:
            output = packed_down(a, weight, packed, large_prefill=large_prefill,
                                 small_split=split_k)
        else:
            output = prefill_linear_nn(a, weight, packed, odd_split=split_k == 3)
        return residual_rms(output, residual, norm_weight, epsilon)

    partials = torch.empty((split_k, m, 2048), device=a.device, dtype=torch.float32)
    output = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    with torch_device_fn.device(a.device):
        if down:
            _down_partial_kernel[(16, split_k)](
                a, packed, partials, m, SPLIT=split_k,
                num_warps=8, num_stages=3,
            )
        else:
            _projection_odd_partial[(32, split_k)](
                a, packed, partials, m, 2048, 2048, 64, 64, split_k,
                num_warps=8, num_stages=3,
            )
        _reduce_residual_rms[(m,)](
            partials, residual, norm_weight, output, residual_out, m,
            SPLIT=split_k, EPS=epsilon, num_warps=8,
        )
    return output, residual_out
