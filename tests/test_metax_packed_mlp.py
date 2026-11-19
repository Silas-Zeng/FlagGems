# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def ops():
    return importlib.import_module("flag_gems.runtime.backend._metax.ops.packed_mlp")


@pytest.fixture(scope="module")
def weights(ops):
    torch.manual_seed(20261103)
    gate = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(2048, 6144, device="cuda", dtype=torch.bfloat16)
    return gate, down, ops.pack_mlp_weight(gate, interleave=True), ops.pack_mlp_weight(down)


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("n,k", [(128, 64), (256, 128)])
def test_weight_pack_is_lossless(ops, interleave, n, k):
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    packed = ops.pack_mlp_weight(w, interleave=interleave)
    unpacked = packed.permute(0, 2, 1, 3).contiguous().view(n, k)
    expected = torch.stack((w[:n // 2], w[n // 2:]), dim=1).view(n, k) if interleave else w
    assert torch.equal(unpacked, expected)


@pytest.mark.parametrize("m", [1, 8, 16, 32, 33, 48, 64, 65, 128, 129, 257, 512, 1025, 2047, 2048])
def test_packed_mlp_vendor(ops, weights, m):
    gate, down, pg, pd = weights
    a = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16) * 0.01
    actual_gate = ops.packed_gate_up_silu(a, gate, pg)
    raw = F.linear(a, gate)
    expected_gate = (F.silu(raw[:, :6144].float()) * raw[:, 6144:].float()).to(a.dtype)
    torch.testing.assert_close(actual_gate, expected_gate, atol=0.01, rtol=0.03)
    actual = ops.packed_down(actual_gate, down, pd)
    torch.testing.assert_close(actual, F.linear(actual_gate, down), atol=0.1, rtol=0.03)


@pytest.mark.parametrize("m", [8, 33, 64, 129, 512, 2048])
def test_packed_mlp_graph_and_weight_reload(ops, m):
    gate = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.01
    down = torch.randn(2048, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
    pg, pd = ops.pack_mlp_weight(gate, interleave=True), ops.pack_mlp_weight(down)
    a = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        ops.packed_down(ops.packed_gate_up_silu(a, gate, pg), down, pd)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = ops.packed_down(ops.packed_gate_up_silu(a, gate, pg), down, pd)
    for _ in range(2):
        a.normal_()
        gate.normal_(std=0.01)
        down.normal_(std=0.01)
        ops.pack_mlp_weight(gate, interleave=True, out=pg)
        ops.pack_mlp_weight(down, out=pd)
        graph.replay()
        raw = F.linear(a, gate)
        hidden = (F.silu(raw[:, :6144].float()) * raw[:, 6144:].float()).to(a.dtype)
        expected = F.linear(hidden, down)
        torch.testing.assert_close(result, expected, atol=0.002, rtol=0.03)


def test_packed_mlp_custom_op_compilation(ops, weights, monkeypatch):
    from vllm_fl.patches import metax_packed_mlp as patch
    monkeypatch.setattr(patch, "_small_split", 4)
    from vllm_fl.patches.metax_packed_mlp import _down, _gate_up

    gate, down, pg, pd = weights
    a = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16) * 0.01
    compiled = torch.compile(
        lambda x: _down(_gate_up(x, gate, pg), down, pd), fullgraph=True
    )
    actual = compiled(a)
    expected = ops.packed_down(ops.packed_gate_up_silu(a, gate, pg), down, pd)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 32, 64, 128, 129, 512, 1025, 2048])
def test_gate_matches_existing_compiled_mlp_exactly(ops, weights, m):
    mm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    gate, _, pg, _ = weights
    a = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    raw = torch.empty(m, 12288, device="cuda", dtype=a.dtype)
    mm.mm_out(a, gate.t(), out=raw)
    activation = torch.compile(
        lambda x: F.silu(x[:, :6144]) * x[:, 6144:], fullgraph=True
    )
    expected = activation(raw)
    actual = ops.packed_gate_up_silu(a, gate, pg)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
