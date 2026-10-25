# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from flag_gems.runtime.backend._metax.ops.fused_gate_up_silu import (
    fused_gate_up_silu,
)


@pytest.mark.parametrize("M", [1, 32, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_gate_up_silu_matches_reference(M):
    torch.manual_seed(0)
    a = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16)

    output = fused_gate_up_silu(a, weight)
    gate_up = torch.mm(a, weight.t())
    reference = F.silu(gate_up[:, :6144].float()) * gate_up[:, 6144:].float()

    torch.testing.assert_close(
        output.float(), reference, rtol=0.03, atol=0.1
    )
