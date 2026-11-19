import pytest
import torch
import triton
from types import SimpleNamespace
from vllm import _custom_ops as vendor
from flag_gems.runtime.backend._metax.ops.rope_cache import rope_qk_cache


def test_install_compatible_llama_layer(monkeypatch):
    from vllm_fl.patches import metax_rope_cache as patch
    from vllm.model_executor.models.llama import LlamaAttention
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backend import AttentionType
    module=object.__new__(LlamaAttention)
    torch.nn.Module.__init__(module)
    module.num_heads,module.num_kv_heads,module.head_dim=16,2,128
    rope=object.__new__(RotaryEmbeddingFL)
    torch.nn.Module.__init__(rope)
    rope.head_size=rope.rotary_dim=128
    rope.is_neox_style=True
    module.rotary_emb=rope
    module.attn=SimpleNamespace(impl=object.__new__(FlashAttentionImpl),
        attn_type=AttentionType.DECODER,kv_cache_dtype='auto',calculate_kv_scales=False,
        query_quant=None,kv_sharing_target_layer_name=None)
    monkeypatch.setenv('VLLM_FL_METAX_ROPE_CACHE','1')
    monkeypatch.setattr(LlamaAttention,'forward',LlamaAttention.forward)
    monkeypatch.setattr(LlamaAttention,'_metax_rope_cache_installed',False,raising=False)
    assert patch.install(module)==1
    assert module._metax_rope_cache_enabled and module.attn._metax_rope_cache_enabled
    assert module.forward.__func__ is patch._forward


@pytest.mark.parametrize("m,actual", [(1,1),(64,61),(64,64),(1985,1981)])
@pytest.mark.parametrize("hnd", [False,True,"page_head_kv"])
def test_rope_and_cache_equal_and_graph(m,actual,hnd):
    torch.manual_seed(20261103)
    x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)
    q,k,v=x[:,:2048],x[:,2048:2304],x[:,2304:]
    pos=torch.arange(m,device='cuda',dtype=torch.int64)
    cache=torch.randn(4096,128,device='cuda',dtype=q.dtype)
    blocks,bs=(actual+15)//16+3,16
    def allocate():
        if hnd == "page_head_kv":
            raw=torch.randn(blocks,2,2,bs,128,device='cuda',dtype=q.dtype)
            return raw[:,:,0].permute(0,2,1,3)
        raw=torch.randn(blocks,2,bs,128,device='cuda',dtype=q.dtype) if hnd else torch.randn(blocks,bs,2,128,device='cuda',dtype=q.dtype)
        return raw.permute(0,2,1,3) if hnd else raw
    kc,vc=allocate(),allocate()
    rk,rv=kc.clone(memory_format=torch.preserve_format),vc.clone(memory_format=torch.preserve_format)
    slots=torch.randperm(blocks*bs,device='cuda',dtype=torch.int64)[:actual].contiguous()
    slots[::11]=-1
    out=torch.empty(m,2304,device='cuda',dtype=q.dtype)
    def call():
        return rope_qk_cache(pos,q,k,v,cache,kc,vc,slots)
    for _ in range(3):call()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=call()
    scale=torch.ones((),device='cuda')
    for _ in range(2):
        x.normal_();pos.add_(1)
        graph.replay()
        oq,ok=q.clone(),k.clone()
        vendor.rotary_embedding(pos,oq,ok,128,cache,True)
        vendor.reshape_and_cache_flash(ok.view(m,2,128),v.view(m,2,128),rk,rv,
                                       slots,'auto',scale,scale)
        assert torch.equal(out[:,:2048],oq) and torch.equal(out[:,2048:],ok)
        assert torch.equal(kc,rk) and torch.equal(vc,rv)


def _state(monkeypatch, metadata=True):
    from vllm_fl.patches import metax_rope_cache as patch
    m=64
    x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)
    pos=torch.arange(m,device='cuda',dtype=torch.int64)
    cache=torch.randn(4096,128,device='cuda',dtype=x.dtype)
    kv=torch.randn(2,32,16,2,128,device='cuda',dtype=x.dtype)
    slots=torch.randperm(512,device='cuda',dtype=torch.int64)[:61].contiguous()
    slots[::7]=-1
    layer=SimpleNamespace(kv_cache=kv,kv_cache_dtype='auto',
        _metax_rope_cache_enabled=True,kv_sharing_target_layer_name=None,
        calculate_kv_scales=False)
    ctx=SimpleNamespace(attn_metadata={'L':SimpleNamespace(slot_mapping=slots)} if metadata else None,
                        no_compile_layers={'L':layer})
    monkeypatch.setattr(patch,'get_forward_context',lambda:ctx)
    return patch,ctx,x,pos,cache,kv,slots


@pytest.mark.parametrize('piecewise',[False,True])
def test_framework_marker_graph_and_cache(monkeypatch,piecewise):
    patch,ctx,x,pos,cache,kv,slots=_state(monkeypatch)
    q,k,v=x[:,:2048],x[:,2048:2304],x[:,2304:]
    compiled=torch.compile(lambda p,q,k,v,c:patch._rope_cache(p,q,k,v,c,'L'),fullgraph=True)
    for _ in range(3):
        compiled(pos,q,k,v,cache)
        assert patch.consume_marker('L')
    expected=kv.clone()
    scale=torch.ones((),device='cuda')
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output=compiled(pos,q,k,v,cache)
        if not piecewise:
            assert patch.consume_marker('L')
    if piecewise: assert patch.consume_marker('L')
    for _ in range(2):
        x.normal_();pos.add_(1)
        graph.replay()
        oq,ok=q.clone(),k.clone()
        vendor.rotary_embedding(pos,oq,ok,128,cache,True)
        vendor.reshape_and_cache_flash(ok.view(64,2,128),v.view(64,2,128),
            expected[0],expected[1],slots,'auto',scale,scale)
        assert torch.equal(output[:,:2048],oq) and torch.equal(output[:,2048:],ok)
        assert torch.equal(kv,expected)
        # On piecewise prefix replay, the CPU marker is absent. The existing
        # update remains enabled and writes the same cache values again.
        assert not patch.consume_marker('L')
        if piecewise:
            vendor.reshape_and_cache_flash(output[:,2048:].view(64,2,128),v.view(64,2,128),
                kv[0],kv[1],slots,'auto',scale,scale)
            assert torch.equal(kv,expected)


@pytest.mark.parametrize('reason',['profiling','invalid_slots','disabled'])
def test_framework_fallback_keeps_original_update(monkeypatch,reason):
    from flag_gems.runtime.backend._metax.ops.rope_qk import rope_qk
    patch,ctx,x,pos,cache,kv,slots=_state(monkeypatch,metadata=reason!='profiling')
    if reason=='invalid_slots':ctx.attn_metadata['L'].slot_mapping=slots.float()
    if reason=='disabled':ctx.no_compile_layers['L']._metax_rope_cache_enabled=False
    before=kv.clone()
    q,k=x[:,:2048],x[:,2048:2304]
    result=patch._rope_cache(pos,q,k,x[:,2304:],cache,'L')
    assert torch.equal(result,rope_qk(pos,q,k,cache))
    assert torch.equal(kv,before)
    assert not patch.consume_marker('L')


def test_flash_attention_consumes_marker_and_retains_fallback(monkeypatch):
    from vllm_fl.patches import metax_rope_cache as patch
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn as backend
    from flash_attn import flash_attn_with_kvcache
    from vllm.v1.attention.backend import AttentionType
    m=3
    x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)
    q,k,v=x[:,:2048],x[:,2048:2304],x[:,2304:]
    pos=torch.arange(m,device='cuda',dtype=torch.int64)
    cache=torch.randn(4096,128,device='cuda',dtype=x.dtype)
    kv=torch.randn(2,m*8,16,2,128,device='cuda',dtype=x.dtype)
    table=torch.arange(m*8,device='cuda',dtype=torch.int32).view(m,8)
    lengths=torch.full((m,),128,device='cuda',dtype=torch.int32)
    slots=torch.arange(m,device='cuda',dtype=torch.int64)*128+127
    scale=torch.ones((),device='cuda')
    layer=SimpleNamespace(kv_cache=kv,kv_cache_dtype='auto',layer_name='L',
        _metax_rope_cache_enabled=True,kv_sharing_target_layer_name=None,
        calculate_kv_scales=False,_q_scale=scale,_k_scale=scale,_v_scale=scale)
    starts=torch.arange(m+1,device='cuda',dtype=torch.int32)
    meta=SimpleNamespace(slot_mapping=slots,num_actual_tokens=m,use_cascade=False,
        query_start_loc=starts,seq_lens=lengths,max_query_len=1,max_seq_len=128,
        block_table=table,scheduler_metadata=None,num_decode_tokens=m,num_prefills=0,
        num_decodes=m,decode_block_table=table,decode_seq_lens=lengths)
    ctx=SimpleNamespace(attn_metadata={'L':meta},no_compile_layers={'L':layer})
    monkeypatch.setattr(patch,'get_forward_context',lambda:ctx)
    impl=backend.FlashAttentionImpl(16,128,128**-.5,2,None,None,'auto',attn_type=AttentionType.DECODER)
    impl.dcp_world_size=1
    copied=[]
    original=backend.reshape_and_cache_flash
    def spy(*args,**kwargs):
        copied.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(backend,'reshape_and_cache_flash',spy)
    out=torch.empty(m,16,128,device='cuda',dtype=x.dtype)
    def call():
        rotated=patch._rope_cache(pos,q,k,v,cache,'L')
        impl.forward(layer,rotated[:,:2048].view(m,16,128),rotated[:,2048:].view(m,2,128),
                     v.view(m,2,128),kv,meta,output=out)
    for _ in range(3):call()
    assert not copied
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    expected=kv.clone()
    for _ in range(2):
        x.normal_();pos.add_(1)
        graph.replay()
        rq,rk=q.clone(),k.clone()
        vendor.rotary_embedding(pos,rq,rk,128,cache,True)
        vendor.reshape_and_cache_flash(rk.view(m,2,128),v.view(m,2,128),expected[0],expected[1],
                                       slots,'auto',scale,scale)
        ref=flash_attn_with_kvcache(rq.view(m,1,16,128),expected[0],expected[1],
              block_table=table,cache_seqlens=lengths,softmax_scale=128**-.5,causal=True,
              window_size=(-1,-1),alibi_slopes=None,softcap=0.,s_aux=None).view_as(out)
        assert torch.equal(kv,expected) and torch.equal(out,ref)
    assert not copied
    impl.forward(layer,rq.view(m,16,128),rk.view(m,2,128),v.view(m,2,128),kv,meta,output=out)
    assert len(copied)==1
