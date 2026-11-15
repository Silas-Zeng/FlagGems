// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include "flash_parameter.h"
#include "kernel_traits.h"
struct RegularSplitParams:mcFlashAttn::Flash_fwd_params {int split_index;};
#include "prefill_regular_splitkv.hpp"
using mcFlashAttn::Flash_fwd_params;
template<int BM,int NW>
using Traits=Flash_fwd_kernel_traits<128,BM,64,NW,true,true,mctlass::bfloat16_t,128>;
template<int BM,int NW>
__global__ void regular_splitkv_prefill_probe(RegularSplitParams p){
 int mb=(p.seqlen_q+BM-1)/BM-1-blockIdx.y;
 p.split_index=blockIdx.z;
 p.o_ptr=(float*)p.o_ptr+size_t(p.split_index)*p.seqlen_q*2048;
 p.softmax_lse_ptr=(float*)p.softmax_lse_ptr+size_t(p.split_index)*p.seqlen_q*16;
 flash::xcore1000::compute_attn_regular_splitkv<Traits<BM,NW>,false,true,false,false,false,false,true,false,false>(p,0,blockIdx.x,mb);
}
extern "C" int flag_gems_c500_prefill(void*q,void*k,void*v,void*out,void*cuq,void*cuk,void*lse,int m,int n,int qs,int kr,int kh,int order,void*stream){
 RegularSplitParams p{};
 p.q_ptr=q;p.k_ptr=k;p.v_ptr=v;p.o_ptr=out;p.softmax_lse_ptr=lse;
 p.b=1;p.h=16;p.h_k=2;p.h_h_k_ratio=8;p.ngroups=8;
 p.q_row_stride=qs;p.o_row_stride=2048;p.q_head_stride=p.o_head_stride=128;
 p.q_batch_stride=m*qs;p.o_batch_stride=m*2048;
 p.k_row_stride=p.v_row_stride=kr;p.k_head_stride=p.v_head_stride=kh;
 p.k_batch_stride=p.v_batch_stride=n*256;
 p.seqlen_q=m;p.seqlen_q_rounded=(m+127)/128*128;p.seqlen_k=n;p.seqlen_k_rounded=(n+63)/64*64;p.total_q=m;
 p.d=p.d_rounded=p.d_value=p.d_value_rounded=128;
 p.cu_seqlens_q=nullptr;p.cu_seqlens_k=nullptr;p.is_seqlens_k_cumulative=false;
 p.scale_softmax=0.08838834764831845f;p.scale_softmax_log2=p.scale_softmax*1.4426950408889634f;
 p.p_dropout=1.f;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
 p.window_size_left=-1;p.window_size_right=0;p.is_causal=true;p.is_bf16=true;p.num_splits=order;
 auto s=(cudaStream_t)stream;
 regular_splitkv_prefill_probe<128,4><<<dim3(16,(m+127)/128,order),256,32768,s>>>(p);

 return int(cudaGetLastError());
}
