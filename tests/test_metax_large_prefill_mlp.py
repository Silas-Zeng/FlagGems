# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def operands():
    ops = importlib.import_module("flag_gems.runtime.backend._metax.ops.packed_mlp")
    torch.manual_seed(20261103)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * .01
    return ops, weight, ops.pack_mlp_weight(weight, interleave=True)


def reference(a, weight):
    raw = F.linear(a, weight)
    return (F.silu(raw[:, :6144].float()) * raw[:, 6144:].float()).to(a.dtype)


@pytest.mark.parametrize("m", [64, 512, 1024, 1025, 1536, 1985, 2047, 2048])
def test_vendor_and_fallback(operands, m):
    ops, weight, packed = operands
    a = torch.randn(m, 2048, device="cuda", dtype=weight.dtype)
    result = ops.packed_gate_up_silu(a, weight, packed, large_prefill=True)
    torch.testing.assert_close(result, reference(a, weight), atol=.01, rtol=.03)
    if m < 1024:
        assert torch.equal(result, ops.packed_gate_up_silu(a, weight, packed))


def test_graph_input_and_weight_refresh(operands):
    ops, weight, packed = operands
    a = torch.randn(1985, 2048, device="cuda", dtype=weight.dtype)
    for _ in range(3):
        ops.packed_gate_up_silu(a, weight, packed, large_prefill=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ops.packed_gate_up_silu(a, weight, packed, large_prefill=True)
    ptr = packed.data_ptr()
    for _ in range(2):
        a.normal_()
        weight.normal_(std=.01)
        ops.pack_mlp_weight(weight, interleave=True, out=packed)
        graph.replay()
        assert packed.data_ptr() == ptr
        torch.testing.assert_close(output, reference(a, weight), atol=.01, rtol=.03)


def test_framework_custom_op(operands, monkeypatch):
    from vllm_fl.patches import metax_packed_mlp as patch

    ops, weight, packed = operands
    monkeypatch.setattr(patch, "_large_prefill", True)
    a = torch.randn(1025, 2048, device="cuda", dtype=weight.dtype)
    compiled = torch.compile(lambda x: patch._gate_up(x, weight, packed), fullgraph=True)
    actual = compiled(a)
    torch.testing.assert_close(actual, ops.packed_gate_up_silu(
        a, weight, packed, large_prefill=True), atol=0, rtol=0)
    torch.testing.assert_close(actual, reference(a, weight), atol=.01, rtol=.03)
