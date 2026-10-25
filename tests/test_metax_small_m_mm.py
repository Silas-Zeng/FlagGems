# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch


@pytest.mark.parametrize(
    ("M", "N", "K", "expected"),
    [
        (1, 2560, 2048, (32, 32, 64, 4, 4)),
        (32, 2560, 2048, (32, 32, 64, 4, 4)),
        (64, 2560, 2048, (64, 32, 128, 8, 2)),
        (32, 12288, 2048, (64, 64, 64, 4, 4)),
        (64, 12288, 2048, (64, 128, 64, 4, 3)),
        (32, 2048, 6144, (32, 32, 128, 4, 2)),
        (64, 2048, 6144, (64, 32, 128, 4, 2)),
        (32, 2048, 2048, (32, 32, 64, 4, 4)),
        (64, 2048, 2048, (64, 32, 128, 4, 2)),
    ],
)
def test_small_m_nt_config(M, N, K, expected):
    mm = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.mm"
    )
    assert mm._small_m_nt_config(M, N, K) == expected
    assert mm._small_m_nt_config(0, N, K) is None
    assert mm._small_m_nt_config(65, N, K) is None


@pytest.mark.parametrize("M", [1, 4, 16, 32, 64])
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


@pytest.mark.parametrize(
    ("M", "N", "K", "expected"),
    [
        (1024, 2560, 2048, (128, 256, 32, 8, 2, 16)),
        (2048, 2560, 2048, (128, 256, 32, 8, 2, 16)),
        (1536, 12288, 2048, (128, 128, 64, 8, 2, 16)),
        (2048, 12288, 2048, (128, 128, 64, 8, 2, 16)),
        (1536, 2048, 6144, (128, 128, 64, 8, 2, 16)),
        (2048, 2048, 6144, (128, 128, 64, 8, 2, 16)),
        (1536, 2048, 2048, (128, 64, 64, 8, 2, 16)),
        (2048, 2048, 2048, (128, 64, 64, 8, 2, 16)),
    ],
)
def test_prefill_nt_config(M, N, K, expected):
    mm = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.mm"
    )
    assert mm._prefill_nt_config(M, N, K) == expected


def test_prefill_nt_config_rejects_unsupported_shapes():
    mm = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.mm"
    )
    assert mm._prefill_nt_config(1023, 2560, 2048) is None
    assert mm._prefill_nt_config(2049, 2560, 2048) is None
    assert mm._prefill_nt_config(2048, 4096, 2048) is None
    assert mm._prefill_nt_config(2048, 2560, 4096) is None


@pytest.mark.parametrize("M", [1024, 2048])
@pytest.mark.parametrize(
    "K,N", [(2048, 2560), (2048, 12288), (6144, 2048), (2048, 2048)]
)
@pytest.mark.parametrize("use_mm_out", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_prefill_nt_matches_torch_mm(M, K, N, use_mm_out):
    mm = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.mm"
    )
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    b = weight.t()
    reference = torch.mm(a, b)

    if use_mm_out:
        output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        mm.mm_out(a, b, out=output)
    else:
        output = mm.mm(a, b)

    torch.testing.assert_close(output, reference, rtol=0.03, atol=0.1)
