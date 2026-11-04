# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch
import torch.nn.functional as F

ops = importlib.import_module("flag_gems.runtime.backend._metax.ops.prefill_linear")
mm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")


@pytest.mark.parametrize("n", [2048, 2560])
@pytest.mark.parametrize("m", [1, 32, 64, 1024, 1536, 1985, 2048])
def test_prefill_linear_matches_current_and_vendor(m, n):
    a = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16) * .01
    w = torch.randn(n, 2048, device="cuda", dtype=torch.bfloat16) * .01
    result = ops.prefill_linear_nn(a, w, w.t().contiguous())
    torch.testing.assert_close(result, mm.mm(a, w.t()), atol=0, rtol=0)
    torch.testing.assert_close(result, F.linear(a, w), atol=.0002, rtol=.03)


@pytest.mark.parametrize("m,n", [(64, 2560), (1025, 2560), (2048, 2048)])
def test_prefill_linear_graph_and_inplace_weight_refresh(m, n):
    a = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16) * .01
    w = torch.randn(n, 2048, device="cuda", dtype=torch.bfloat16) * .01
    p = w.t().contiguous()
    pointer = p.data_ptr()
    for _ in range(3):
        ops.prefill_linear_nn(a, w, p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = ops.prefill_linear_nn(a, w, p)
    for _ in range(2):
        a.normal_(std=.01)
        w.normal_(std=.01)
        p.copy_(w.t())
        graph.replay()
        assert p.data_ptr() == pointer
        torch.testing.assert_close(result, mm.mm(a, w.t()), atol=0, rtol=0)


def test_prefill_custom_op_fullgraph():
    from vllm_fl.patches.metax_prefill_nn import _linear
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) * .01
    w = torch.randn(2560, 2048, device="cuda", dtype=torch.bfloat16) * .01
    p = w.t().contiguous()
    compiled = torch.compile(lambda x: _linear(x, w, p), fullgraph=True)
    torch.testing.assert_close(compiled(a), mm.mm(a, w.t()), atol=0, rtol=0)


def test_install_and_reload_preserve_cache_pointers(monkeypatch):
    from vllm.model_executor.layers.linear import QKVParallelLinear, UnquantizedLinearMethod
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from vllm_fl.patches import metax_prefill_nn as patch
    layer = object.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(torch.randn(2560, 2048, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    layer.bias = None
    layer.tp_size = 1
    layer.quant_method = UnquantizedLinearMethod()
    model = torch.nn.Module()
    model.add_module("qkv", layer)
    monkeypatch.setenv("VLLM_FL_METAX_PREFILL_NN", "1")
    monkeypatch.setenv("VLLM_FL_METAX_PROJECTION_PACK", "0")
    monkeypatch.setattr(UnquantizedLinearMethod, "apply", UnquantizedLinearMethod.apply)
    monkeypatch.setattr(UnquantizedLinearMethod, "_metax_prefill_nn_installed", False, raising=False)
    monkeypatch.setattr(LlamaForCausalLM, "load_weights", LlamaForCausalLM.load_weights)
    monkeypatch.setattr(LlamaForCausalLM, "_metax_prefill_nn_reload", False, raising=False)
    assert patch.install(model) == 1
    cache = layer._metax_prefill_nn_weight
    pointer = cache.data_ptr()
    assert "_metax_prefill_nn_weight" not in layer.state_dict()
    def loader(model, weights):
        layer.weight.normal_(std=.01)
        return {"qkv.weight"}
    monkeypatch.setattr(patch, "_original_load_weights", loader)
    assert patch._load_weights(model, []) == {"qkv.weight"}
    assert layer._metax_prefill_nn_weight.data_ptr() == pointer
    torch.testing.assert_close(cache, layer.weight.t(), atol=0, rtol=0)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(layer.quant_method.apply(layer, a), mm.mm(a, layer.weight.t()), atol=0, rtol=0)
