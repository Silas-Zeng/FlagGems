# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch
import torch.nn.functional as F

mlp = importlib.import_module("flag_gems.runtime.backend._metax.ops.packed_mlp")
linear = importlib.import_module("flag_gems.runtime.backend._metax.ops.prefill_linear")


@pytest.mark.parametrize("kind,k", [("down", 6144), ("o", 2048)])
@pytest.mark.parametrize("m", [1, 32, 33, 48, 63, 64, 65, 2048])
def test_vendor_and_other_shapes(kind, k, m):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)*.01
    w = torch.randn(2048, k, device="cuda", dtype=a.dtype)*.01
    p = mlp.pack_mlp_weight(w)
    result = (mlp.packed_down(a, w, p, small_split=6) if kind == "down" else
              linear.prefill_linear_nn(a, w, p, odd_split=True))
    torch.testing.assert_close(result, F.linear(a, w), atol=.0002, rtol=.03)
    if not 33 <= m <= 64:
        expected = mlp.packed_down(a, w, p) if kind == "down" else linear.prefill_linear_nn(a, w, p)
        assert torch.equal(result, expected)


@pytest.mark.parametrize("kind,k", [("down", 6144), ("o", 2048)])
def test_graph_input_and_weight_refresh(kind, k):
    a = torch.randn(48, k, device="cuda", dtype=torch.bfloat16)*.01
    w = torch.randn(2048, k, device="cuda", dtype=a.dtype)*.01
    p = mlp.pack_mlp_weight(w)
    def call():
        return (mlp.packed_down(a, w, p, small_split=6) if kind == "down" else
                linear.prefill_linear_nn(a, w, p, odd_split=True))
    for _ in range(3): call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): result = call()
    for _ in range(2):
        a.normal_(std=.01); w.normal_(std=.01)
        mlp.pack_mlp_weight(w, out=p)
        graph.replay()
        torch.testing.assert_close(result, F.linear(a, w), atol=.0002, rtol=.03)


def test_framework_custom_ops(monkeypatch):
    from vllm_fl.patches import metax_packed_mlp, metax_prefill_nn
    monkeypatch.setattr(metax_packed_mlp, "_small_split", 6)
    monkeypatch.setattr(metax_prefill_nn, "_odd_split", True)
    a = torch.randn(48, 6144, device="cuda", dtype=torch.bfloat16)*.01
    b = torch.randn(48, 2048, device="cuda", dtype=a.dtype)*.01
    wd = torch.randn(2048, 6144, device="cuda", dtype=a.dtype)*.01
    wo = torch.randn(2048, 2048, device="cuda", dtype=a.dtype)*.01
    pd, po = mlp.pack_mlp_weight(wd), mlp.pack_mlp_weight(wo)
    compiled = torch.compile(lambda x,y: (metax_packed_mlp._down(x, wd, pd),
                                          metax_prefill_nn._linear(y, wo, po)), fullgraph=True)
    rd, ro = compiled(a,b)
    torch.testing.assert_close(rd, F.linear(a,wd), atol=.0002, rtol=.03)
    torch.testing.assert_close(ro, F.linear(b,wo), atol=.0002, rtol=.03)
