# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch
from vllm import _custom_ops as vendor
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from flag_gems.runtime.backend._metax.ops.rope_qk import rope_qk


@pytest.fixture(scope="module")
def cache():
    freq = 1/(5000000.**(torch.arange(0, 128, 2, device="cuda").float()/128))
    angles = torch.arange(4096, device="cuda").float()[:, None]*freq[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()


def inputs(m, shared):
    if shared:
        x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
        return x[:, :2048], x[:, 2048:2304]
    return (torch.randn(m, 2056, device="cuda", dtype=torch.bfloat16)[:, :2048],
            torch.randn(m, 264, device="cuda", dtype=torch.bfloat16)[:, :256])


def reference(pos, q, k, cache):
    q, k = q.clone(), k.clone()
    vendor.rotary_embedding(pos.long(), q, k, 128, cache, True)
    return q, k


@pytest.mark.parametrize("m", [0, 1, 8, 32, 64, 65, 1985, 2048])
@pytest.mark.parametrize("shared", [False, True])
def test_vendor_and_layout(cache, m, shared):
    q, k = inputs(m, shared)
    before_q, before_k = q.clone(), k.clone()
    pos = torch.arange(m, device="cuda", dtype=torch.int64)
    out = rope_qk(pos, q, k, cache)
    if m:
        rq, rk = reference(pos, q, k, cache)
        assert torch.equal(out[:, :2048], rq)
        assert torch.equal(out[:, 2048:], rk)
    assert torch.equal(q, before_q) and torch.equal(k, before_k)
    assert out.shape == (m, 2304)


def test_int32_negative_positions(cache):
    q, k = inputs(7, False)
    pos = torch.tensor([-1, -4, 0, 1, 63, 512, 4095], device="cuda", dtype=torch.int32)
    out = rope_qk(pos, q, k, cache)
    rq, rk = reference(pos+torch.where(pos < 0, 4096, 0), q, k, cache)
    assert torch.equal(out[:, :2048], rq) and torch.equal(out[:, 2048:], rk)


def test_graph_inputs_positions_and_cache(cache):
    q, k = inputs(64, True)
    pos = torch.arange(64, device="cuda", dtype=torch.int64)
    c = cache.clone()
    for _ in range(3): rope_qk(pos, q, k, c)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out = rope_qk(pos, q, k, c)
    for _ in range(2):
        q.normal_(); k.normal_(); pos.add_(128)
        c.copy_(cache.roll(1, 0))
        graph.replay()
        rq, rk = reference(pos, q, k, c)
        assert torch.equal(out[:, :2048], rq) and torch.equal(out[:, 2048:], rk)


@pytest.mark.parametrize("plugin_class", [False, True])
def test_module_install_compile_and_fallback(cache, monkeypatch, plugin_class):
    from vllm_fl.patches import metax_fused_rope as patch
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
    layer = object.__new__(RotaryEmbeddingFL if plugin_class else RotaryEmbedding)
    torch.nn.Module.__init__(layer)
    layer.head_size = layer.rotary_dim = 128
    layer.dtype = torch.bfloat16
    layer.is_neox_style = True
    layer.register_buffer("cos_sin_cache", cache, persistent=False)
    layer._forward_method = types.MethodType(RotaryEmbedding.forward_native, layer)
    monkeypatch.setenv("VLLM_FL_METAX_FUSED_ROPE", "1")
    assert patch.install(layer) == 1
    compiled = torch.compile(lambda p,q,k: layer(p,q,k), fullgraph=True)
    q, k = inputs(64, True)
    pos = torch.arange(64, device="cuda", dtype=torch.int64)
    actual = compiled(pos, q, k)
    expected = reference(pos, q, k, cache)
    for a,b in zip(actual, expected): assert torch.equal(a, b)
    assert actual[0].stride(0) == actual[1].stride(0) == 2304
    query_only = torch.compile(lambda p,q: layer(p,q,None)[0], fullgraph=True)
    assert torch.equal(query_only(pos, q), expected[0])


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_attention_and_kv_cache_with_output_views(cache, mode):
    from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.flash_attn_out import (
        flash_attn_kvcache_into, flash_attn_varlen_into,
    )
    b, blocks, block = 3, 8, 16
    m = b if mode == "decode" else b*17
    q, k = inputs(m, True)
    pos = torch.arange(m, device="cuda", dtype=torch.int64)
    output = rope_qk(pos, q, k, cache)
    oq, ok = output[:, :2048].view(m, 16, 128), output[:, 2048:].view(m, 2, 128)
    kc = torch.randn(b*blocks, block, 2, 128, device="cuda", dtype=q.dtype)
    vc = torch.randn_like(kc)
    reference_kc = kc.clone()
    reference_vc = vc.clone()
    value = torch.randn(m, 256, device="cuda", dtype=q.dtype).view(m, 2, 128)
    slots = torch.arange(m, device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda")
    vendor.reshape_and_cache_flash(ok, value, kc, vc, slots, "auto", scale, scale)
    vendor.reshape_and_cache_flash(ok.contiguous(), value, reference_kc, reference_vc,
                                   slots, "auto", scale, scale)
    assert torch.equal(kc, reference_kc) and torch.equal(vc, reference_vc)
    table = torch.arange(b*blocks, device="cuda", dtype=torch.int32).view(b, blocks)
    common = dict(block_table=table, softmax_scale=128**-.5, causal=True,
                  window_size=(-1, -1), alibi_slopes=None, softcap=0., s_aux=None)
    out = torch.empty_like(oq)
    if mode == "decode":
        kwargs = dict(common, cache_seqlens=torch.full((b,), 128, device="cuda", dtype=torch.int32))
        flash_attn_kvcache_into(oq.view(b,1,16,128), kc, vc, out.view(b,1,16,128), **kwargs)
        ref = flash_attn_with_kvcache(oq.contiguous().view(b,1,16,128), kc, vc, **kwargs).view_as(out)
    else:
        kwargs = dict(common, cu_seqlens_q=torch.arange(0,m+1,17,device="cuda",dtype=torch.int32),
                      cu_seqlens_k=torch.arange(0,b*128+1,128,device="cuda",dtype=torch.int32),
                      max_seqlen_q=17, max_seqlen_k=128)
        flash_attn_varlen_into(oq, kc, vc, out, **kwargs)
        ref = flash_attn_varlen_func(oq.contiguous(), kc, vc, **kwargs)
    assert torch.equal(out, ref)
