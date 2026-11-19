# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Coalesced weight reads for small batches and wide BF16 linear layers."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.linear import linear as _generic_linear
from flag_gems.runtime import torch_device_fn


@triton.jit(do_not_specialize=["M"])
def _wide_linear_kernel(
    A, W, O, M, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # M varies for every request admission/completion. Keep only the three tile
    # families specialized instead of compiling a new kernel for each batch.
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + m[:, None] * K + k[None, :]
    wp = W + n[:, None] * K + k[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap, m[:, None] < M, 0.0)
        w = tl.load(wp)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32)
        ap += BK
        wp += BK
    tl.store(
        O + m[:, None] * N + n[None, :],
        acc.to(O.dtype.element_ty),
        m[:, None] < M,
    )


def linear(input, weight, bias=None):
    if (
        bias is None
        and input.ndim >= 1
        and weight.ndim == 2
        and input.dtype == weight.dtype == torch.bfloat16
        and input.device == weight.device
        and input.device.type == "cuda"
        and input.is_contiguous()
        and weight.is_contiguous()
        and input.shape[-1] == weight.shape[1] == 2048
        and weight.shape[0] >= 32768
        and weight.shape[0] % 128 == 0
        and 0 < input.numel() // 2048 <= 64
    ):
        M, K = input.numel() // 2048, 2048
        N = weight.shape[0]
        if M <= 16:
            bm, bn, bk, warps, stages = 16, 32, 64, 4, 3
        elif M <= 32:
            bm, bn, bk, warps, stages = 32, 64, 64, 4, 2
        else:
            bm, bn, bk, warps, stages = 64, 128, 64, 4, 3
        output = torch.empty(
            (*input.shape[:-1], N), dtype=input.dtype, device=input.device
        )
        with torch_device_fn.device(input.device):
            _wide_linear_kernel[(triton.cdiv(N, bn),)](
                input,
                weight,
                output,
                M,
                N,
                K,
                BM=bm,
                BN=bn,
                BK=bk,
                num_warps=warps,
                num_stages=stages,
            )
        return output
    return _generic_linear(input, weight, bias)
