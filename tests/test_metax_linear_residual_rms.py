# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm import _custom_ops as vendor

from flag_gems.runtime.backend._metax.ops.linear_residual_rms import linear_residual_rms
from flag_gems.runtime.backend._metax.ops.packed_mlp import pack_mlp_weight, packed_down
from flag_gems.runtime.backend._metax.ops.prefill_linear import prefill_linear_nn
from flag_gems.runtime.backend._metax.ops.residual_rms import residual_rms


@pytest.fixture(scope="module", params=[True, False])
def operands(request):
    down = request.param
    k = 6144 if down else 2048
    weight = torch.randn(2048, k, device="cuda", dtype=torch.bfloat16) * .01
    gamma = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    return down, weight, pack_mlp_weight(weight), gamma


def reference(a, weight, packed, residual, gamma, down, split):
    if down:
        y = packed_down(a, weight, packed, large_prefill=True, small_split=split)
    else:
        y = prefill_linear_nn(a, weight, packed, odd_split=split == 3)
    return residual_rms(y, residual, gamma, 1e-6)


@pytest.mark.parametrize("m", [0, 1, 32, 33, 48, 64, 65, 1985])
def test_numerics_and_fallback(operands, m):
    down, weight, packed, gamma = operands
    split = 6 if down else 3
    a = torch.randn(m, weight.shape[1], device="cuda", dtype=weight.dtype)
    r = torch.randn(m, 2048, device="cuda", dtype=weight.dtype)
    output, residual = linear_residual_rms(a, weight, packed, r, gamma, 1e-6,
                                          down=down, split_k=split)
    expected, expected_res = reference(a, weight, packed, r, gamma, down, split)
    torch.testing.assert_close(output, expected, atol=.02, rtol=.03)
    assert torch.equal(residual, expected_res)
    if m:
        native, native_res = torch.mm(a, weight.t()), r.clone()
        vendor.fused_add_rms_norm(native, native_res, gamma, 1e-6)
        torch.testing.assert_close(output, native, atol=.02, rtol=.03)


def test_graph_input_and_weight_updates(operands):
    down, weight, packed, gamma = operands
    split = 6 if down else 3
    a = torch.randn(64, weight.shape[1], device="cuda", dtype=weight.dtype)
    r = torch.randn(64, 2048, device="cuda", dtype=weight.dtype)
    def run():
        return linear_residual_rms(a, weight, packed, r, gamma, 1e-6,
                                   down=down, split_k=split)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, residual = run()
    for _ in range(2):
        a.normal_(); r.normal_(); gamma.normal_(); weight.normal_(std=.01)
        pack_mlp_weight(weight, out=packed)
        graph.replay()
        expected, expected_res = reference(a, weight, packed, r, gamma, down, split)
        torch.testing.assert_close(output, expected, atol=.02, rtol=.03)
        assert torch.equal(residual, expected_res)


def test_disabled_odd_splits(operands):
    down, weight, packed, gamma = operands
    split = 4 if down else 0
    a = torch.randn(48, weight.shape[1], device="cuda", dtype=weight.dtype)
    r = torch.randn(48, 2048, device="cuda", dtype=weight.dtype)
    output, residual = linear_residual_rms(a, weight, packed, r, gamma, 1e-6,
                                          down=down, split_k=split)
    expected, expected_res = reference(a, weight, packed, r, gamma, down, split)
    torch.testing.assert_close(output, expected, atol=.02, rtol=.03)
    assert torch.equal(residual, expected_res)


def test_compile_pass_fuses_supported_pairs(operands):
    down, weight, packed, gamma = operands
    from vllm_fl.patches.metax_linear_norm_fusion import LinearNormFusionPass
    from vllm_fl.patches.metax_packed_mlp import _down
    from vllm_fl.patches.metax_prefill_nn import _linear
    from vllm_fl.patches.metax_residual_rms import _norm
    projection = _down if down else _linear
    fusion = LinearNormFusionPass()
    def fn(a, w, p, r, g):
        return _norm(projection(a, w, p), r, g, 1e-6)
    compiled = torch.compile(fn, fullgraph=True,
                              options={"post_grad_custom_post_pass": fusion,
                                       "force_disable_caches": True})
    for m in [64, 48]:
        a = torch.randn(m, weight.shape[1], device="cuda", dtype=weight.dtype)
        r = torch.randn(m, 2048, device="cuda", dtype=weight.dtype)
        result = compiled(a, weight, packed, r, gamma)
        expected = fn(a, weight, packed, r, gamma)
        for actual, ref in zip(result, expected):
            torch.testing.assert_close(actual, ref, atol=.02, rtol=.03)
    assert fusion.matched_count > 0


def test_extra_projection_consumer_is_preserved(operands):
    down, weight, packed, gamma = operands
    from vllm_fl.patches.metax_linear_norm_fusion import LinearNormFusionPass
    from vllm_fl.patches.metax_packed_mlp import _down
    from vllm_fl.patches.metax_prefill_nn import _linear
    from vllm_fl.patches.metax_residual_rms import _norm
    projection = _down if down else _linear
    fusion = LinearNormFusionPass()
    def fn(a, w, p, r, g):
        projected = projection(a, w, p)
        output, residual = _norm(projected, r, g, 1e-6)
        return projected, output, residual
    compiled = torch.compile(fn, fullgraph=True,
                              options={"post_grad_custom_post_pass": fusion,
                                       "force_disable_caches": True})
    a = torch.randn(64, weight.shape[1], device="cuda", dtype=weight.dtype)
    r = torch.randn(64, 2048, device="cuda", dtype=weight.dtype)
    result = compiled(a, weight, packed, r, gamma)
    for actual, ref in zip(result, fn(a, weight, packed, r, gamma)):
        torch.testing.assert_close(actual, ref, atol=0, rtol=0)
    assert fusion.matched_count == 0
