# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import types
import pytest
import torch
from vllm import _custom_ops as vendor
from flag_gems.runtime.backend._metax.ops.residual_rms import residual_rms


@pytest.mark.parametrize('m',[0,1,8,32,64,65,129,1985,2048])
@pytest.mark.parametrize('scale',[.01,1.,100.])
def test_vendor(m,scale):
    x=torch.randn(m,2048,device='cuda',dtype=torch.bfloat16)*scale
    r=torch.randn_like(x)*scale
    w=torch.randn(2048,device='cuda',dtype=x.dtype)
    a,b=x.clone(),r.clone()
    output,res=residual_rms(x,r,w,1e-6)
    if m:
        expected,er=x.clone(),r.clone()
        vendor.fused_add_rms_norm(expected,er,w,1e-6)
        torch.testing.assert_close(output,expected,atol=.02,rtol=.03)
        assert torch.equal(res,er)
    assert torch.equal(x,a) and torch.equal(r,b)


@pytest.mark.parametrize('m',[64,1985])
def test_graph_updates(m):
    x=torch.randn(m,2048,device='cuda',dtype=torch.bfloat16)
    r=torch.randn_like(x)
    w=torch.randn(2048,device='cuda',dtype=x.dtype)
    for _ in range(3):residual_rms(x,r,w,1e-6)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output,res=residual_rms(x,r,w,1e-6)
    for _ in range(2):
        x.normal_();r.normal_();w.normal_()
        graph.replay()
        expected,er=x.clone(),r.clone()
        vendor.fused_add_rms_norm(expected,er,w,1e-6)
        torch.testing.assert_close(output,expected,atol=.02,rtol=.03)
        assert torch.equal(res,er)


@pytest.mark.parametrize('plugin_class',[False,True])
def test_install_compile_and_no_residual_fallback(monkeypatch,plugin_class):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.patches import metax_residual_rms as patch
    layer=object.__new__(RMSNormFL if plugin_class else RMSNorm)
    torch.nn.Module.__init__(layer)
    layer.hidden_size=2048;layer.variance_size_override=None;layer.variance_epsilon=1e-6
    layer.pass_weight_add=layer.pass_weight=True
    layer.weight=torch.nn.Parameter(torch.randn(2048,device='cuda',dtype=torch.bfloat16),requires_grad=False)
    layer._forward_method=types.MethodType(RMSNorm.forward_native,layer)
    monkeypatch.setenv('VLLM_FL_METAX_RESIDUAL_RMS','1')
    assert patch.install(layer)==1
    x=torch.randn(64,2048,device='cuda',dtype=torch.bfloat16)
    r=torch.randn_like(x)
    compiled=torch.compile(lambda a,b:layer(a,b),fullgraph=True)
    output,res=compiled(x,r)
    expected,er=x.clone(),r.clone()
    vendor.fused_add_rms_norm(expected,er,layer.weight,1e-6)
    torch.testing.assert_close(output,expected,atol=.02,rtol=.03)
    assert torch.equal(res,er)
    compiled_single=torch.compile(lambda a:layer(a,None),fullgraph=True)
    single=compiled_single(x)
    v=torch.empty_like(x)
    vendor.rms_norm(v,x,layer.weight,1e-6)
    torch.testing.assert_close(single,v,atol=.02,rtol=.03)
