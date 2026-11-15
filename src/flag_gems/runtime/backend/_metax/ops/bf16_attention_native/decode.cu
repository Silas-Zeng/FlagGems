// SPDX-License-Identifier: Apache-2.0
// Experimental tile launcher for the MetaX BF16 FlashAttention kernel.
// Kernel reference: MetaX-MACA/flashattn (Tri Dao / MetaX copyright in headers).
#include <cuda_runtime.h>
#include "flash_parameter.h"
#include "kernel_traits.h"
#include "decode_kernel.h"
#include "flash_fwd_split_combine_kernel.h"
#include <cmath>
using mcFlashAttn::Flash_fwd_params;
template<int BN,int NW>
using Traits=Flash_fwd_kernel_traits<128,16,BN,NW,true,true,mctlass::bfloat16_t,128>;
template<int BN,int NW,bool HeadFirst=false>
__global__ void flag_gems_c500_decode_tile(Flash_fwd_params p) {
  p.page_block_size=16;
  p.cu_seqlens_q=nullptr;p.seqused_k=nullptr;p.leftpad_k=nullptr;p.knew_ptr=nullptr;p.vnew_ptr=nullptr;
  p.cache_batch_idx=nullptr;p.unpadded_lse=false;
  p.scale_softmax=0.08838834764831845f;
  p.scale_softmax_log2=0.08838834764831845f*1.4426950408889634f;
  p.p_dropout=1.f;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=0.08838834764831845f;
  p.k_batch_stride=p.v_batch_stride=8192;
  p.k_row_stride=p.v_row_stride=128;
  p.k_head_stride=p.v_head_stride=4096;
  p.q_row_stride=p.o_row_stride=128;
  p.q_head_stride=p.o_head_stride=1024;
  p.h=p.h_k=2;p.h_h_k_ratio=1;p.ngroups=8;
  p.seqlen_q=8;p.seqlen_q_rounded=16;
  p.d=p.d_rounded=p.d_value=p.d_value_rounded=128;
  p.is_seqlens_k_cumulative=false;
  p.window_size_left=p.window_size_right=-1;
  p.num_splits=6;
  // Interleave KV heads across batches at B64; retain the original ordering
  // for smaller batches, where the extra memory-level parallelism did not help.
  flash::xcore1000::compute_attn_1rowblock_splitkv_opt_hdim128_kblockMsmall<Traits<BN,NW>,false,false,false,false,true,false,true,false,true>(
      p,blockIdx.y,HeadFirst ? blockIdx.x : blockIdx.z,0,
      HeadFirst ? blockIdx.z : blockIdx.x,p.num_splits);
}
__global__ void flag_gems_c500_combine_tile(Flash_fwd_params p) {
  p.h=p.h_k=2;p.h_h_k_ratio=1;p.ngroups=8;p.seqlen_q=8;p.d=p.d_rounded=p.d_value=p.d_value_rounded=128;p.num_splits=6;
  p.seqlenq_ngroups_swapped=true;p.o_batch_stride=2048;p.o_row_stride=128;p.o_head_stride=1024;p.unpadded_lse=false;
  flash::combine_attn_seqk_parallel<Traits<16,1>,8,3,true,false>(p);
}
template<int BN,int NW>
void launch(Flash_fwd_params &p,cudaStream_t s) {
  constexpr int smem=std::max(2*Traits<BN,NW>::kSmemQSize,std::max(Traits<BN,NW>::kSmemKSize,Traits<BN,NW>::kSmemVSize));
  if (p.b==64) {
    flag_gems_c500_decode_tile<BN,NW,true><<<dim3(p.h,p.b,p.num_splits),NW*64,smem,s>>>(p);
  } else {
    flag_gems_c500_decode_tile<BN,NW><<<dim3(p.num_splits,p.b,p.h),NW*64,smem,s>>>(p);
  }
}
extern "C" int flag_gems_c500_decode(void* q,void* k,void* v,void* out,void* table,void* lens,void* accum,void* lsepart,void* lse,int batch,int query_batch_stride,int table_stride,void* stream) {
  constexpr int page=16,splits=6;
  const int seq=table_stride*page;
  Flash_fwd_params p{};
  p.q_ptr=q;p.k_ptr=k;p.v_ptr=v;p.o_ptr=out;
  p.q_batch_stride=query_batch_stride;p.q_row_stride=128;p.q_head_stride=1024;
  p.k_batch_stride=p.v_batch_stride=page*4*128;
  p.k_row_stride=p.v_row_stride=128;p.k_head_stride=p.v_head_stride=page*2*128;
  p.o_batch_stride=2048;p.o_row_stride=128;p.o_head_stride=1024;
  p.b=batch;p.h=p.h_k=2;p.h_h_k_ratio=1;p.ngroups=8;
  p.seqlen_q=8;p.seqlen_q_rounded=16;p.seqlen_k=seq;p.seqlen_k_rounded=(seq+15)/16*16;
  p.d=p.d_rounded=p.d_value=p.d_value_rounded=128;p.total_q=batch*8;
  p.scale_softmax=1.f/sqrtf(128.f);p.scale_softmax_log2=p.scale_softmax*1.4426950408889634f;
  p.cu_seqlens_k=(int*)lens;p.is_seqlens_k_cumulative=false;
  p.block_table=(int*)table;p.block_table_batch_stride=table_stride;p.page_block_size=page;
  p.p_dropout=1.f;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
  p.window_size_left=p.window_size_right=-1;p.is_bf16=true;p.is_causal=false;
  p.num_splits=splits;p.oaccum_ptr=accum;p.softmax_lseaccum_ptr=lsepart;p.softmax_lse_ptr=lse;
  p.seqlenq_ngroups_swapped=true;
  auto s=(cudaStream_t)stream;

  launch<32,1>(p,s);

  flag_gems_c500_combine_tile<<<dim3((batch*2*8+7)/8),256,0,s>>>(p);
  return int(cudaGetLastError());
}
