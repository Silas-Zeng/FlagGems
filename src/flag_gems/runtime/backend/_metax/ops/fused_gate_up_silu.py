# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _fused_gate_up_silu_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    gate_ptrs = B + rn[:, None] * K + rk[None, :]
    up_ptrs = B + (rn + N)[:, None] * K + rk[None, :]
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
        gate_acc = tl.dot(
            a,
            tl.trans(gate),
            gate_acc,
            out_dtype=tl.float32,
            allow_tf32=False,
        )
        up_acc = tl.dot(
            a,
            tl.trans(up),
            up_acc,
            out_dtype=tl.float32,
            allow_tf32=False,
        )
        a_ptrs += BLOCK_K
        gate_ptrs += BLOCK_K
        up_ptrs += BLOCK_K

    silu_gate = gate_acc / (1.0 + tl.exp(-gate_acc))
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    tl.store(
        c_ptrs,
        silu_gate * up_acc,
        mask=(rm < M)[:, None] & (rn < N)[None, :],
    )


def fused_gate_up_silu(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """MiniCPM BF16 Gate/Up GEMM with fused SwiGLU epilogue."""
    assert a.ndim == 2 and weight.ndim == 2
    M, K = a.shape
    N2, weight_K = weight.shape
    assert N2 % 2 == 0
    N = N2 // 2
    assert (K, N, weight_K) == (2048, 6144, 2048)
    assert 1 <= M <= 64
    assert a.dtype == weight.dtype == torch.bfloat16
    assert a.is_contiguous() and weight.is_contiguous()

    block_m = 32 if M <= 32 else 64
    block_n = 64
    block_k = 64
    num_warps = 4
    num_stages = 4 if M <= 32 else 3
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(N, block_n),)
    with torch_device_fn.device(a.device):
        _fused_gate_up_silu_kernel[grid](
            a,
            weight,
            out,
            M,
            N,
            K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
