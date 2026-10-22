# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch


@pytest.mark.parametrize(
    ("N", "K", "expected"),
    [
        (2560, 2048, (64, 32, 64, 4, 4)),
        (12288, 2048, (64, 64, 64, 4, 2)),
        (2048, 6144, (32, 64, 64, 4, 4)),
        (2048, 2048, (32, 64, 64, 4, 4)),
    ],
)
def test_small_m_nt_config(N, K, expected):
    mm = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.mm"
    )
    assert mm._small_m_nt_config(64, N, K) == expected
    assert mm._small_m_nt_config(31, N, K) is None
    assert mm._small_m_nt_config(65, N, K) is None


@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize(
    "K,N", [(2048, 2560), (2048, 12288), (6144, 2048), (2048, 2048)]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_small_m_nt_matches_torch_mm(M, K, N):
    mm = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.mm"
    )
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    mm.mm_out(a, weight.t(), out=output)
    reference = torch.mm(a, weight.t())
    torch.testing.assert_close(output, reference, rtol=0.03, atol=0.1)
