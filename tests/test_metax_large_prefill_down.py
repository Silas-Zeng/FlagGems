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
    w = torch.randn(2048, 6144, device="cuda", dtype=torch.bfloat16)*.01
    return ops, w, ops.pack_mlp_weight(w)


@pytest.mark.parametrize("m", [64, 1536, 1792, 1793, 1985, 2047, 2048])
def test_vendor_and_fallback(operands, m):
    ops, weight, packed = operands
    a = torch.randn(m, 6144, device="cuda", dtype=weight.dtype)
    result = ops.packed_down(a, weight, packed, large_prefill=True)
    torch.testing.assert_close(result, F.linear(a, weight), atol=.01, rtol=.03)
    if m < 1792:
        assert torch.equal(result, ops.packed_down(a, weight, packed))


def test_noncontiguous_fallback(operands):
    ops, weight, packed = operands
    a = torch.randn(1793, 12288, device="cuda", dtype=weight.dtype)[:, ::2]
    result = ops.packed_down(a, weight, packed, large_prefill=True)
    torch.testing.assert_close(result, F.linear(a, weight), atol=.01, rtol=.03)


def test_graph_input_weight_refresh(operands):
    ops, weight, packed = operands
    a = torch.randn(1985, 6144, device="cuda", dtype=weight.dtype)
    for _ in range(3): ops.packed_down(a, weight, packed, large_prefill=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ops.packed_down(a, weight, packed, large_prefill=True)
    ptr = packed.data_ptr()
    for _ in range(2):
        a.normal_()
        weight.normal_(std=.01)
        ops.pack_mlp_weight(weight, out=packed)
        graph.replay()
        assert packed.data_ptr() == ptr
        torch.testing.assert_close(output, F.linear(a, weight), atol=.01, rtol=.03)


def test_compiled_mlp_chain(operands, monkeypatch):
    from vllm_fl.patches import metax_packed_mlp as patch

    ops, down, pd = operands
    monkeypatch.setattr(patch, "_large_prefill", True)
    monkeypatch.setattr(patch, "_large_down_prefill", True)
    gate = torch.randn(12288, 2048, device="cuda", dtype=down.dtype)*.01
    pg = ops.pack_mlp_weight(gate, interleave=True)
    a = torch.randn(1793, 2048, device="cuda", dtype=down.dtype)
    compiled = torch.compile(lambda x: patch._down(
        patch._gate_up(x, gate, pg), down, pd), fullgraph=True)
    result = compiled(a)
    raw = F.linear(a, gate)
    hidden = (F.silu(raw[:, :6144].float())*raw[:, 6144:].float()).to(a.dtype)
    torch.testing.assert_close(result, F.linear(hidden, down), atol=.01, rtol=.03)
