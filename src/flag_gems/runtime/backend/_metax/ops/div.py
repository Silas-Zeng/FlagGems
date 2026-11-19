# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-broadcast FP32 division without generic multidimensional indexing."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.div import true_divide_ as _generic_true_divide_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_lang_extension import div_rn


@triton.jit
def _row_broadcast_div_kernel(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(A + row * N + col, col < N, 0.0)
    divisor = tl.load(B + row)
    # Correctly rounded division also handles zero, infinity and NaN.
    result = div_rn(value, divisor)
    tl.store(A + row * N + col, result, col < N)


def true_divide_(A, B):
    if (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and A.ndim == B.ndim == 2
        and A.dtype == B.dtype == torch.float32
        and A.device == B.device
        and A.device.type == "cuda"
        and A.is_contiguous()
        and B.is_contiguous()
        and B.shape == (A.shape[0], 1)
        and A.shape[0] > 0
        and A.shape[1] >= 32768
    ):
        # B must not alias the modified A: all column tiles share its row value.
        if A.untyped_storage().data_ptr() != B.untyped_storage().data_ptr():
            with torch_device_fn.device(A.device):
                _row_broadcast_div_kernel[
                    (A.shape[0], triton.cdiv(A.shape[1], 2048))
                ](
                    A,
                    B,
                    A.shape[1],
                    BLOCK=2048,
                    num_warps=4,
                )
            return A
    return _generic_true_divide_(A, B)
