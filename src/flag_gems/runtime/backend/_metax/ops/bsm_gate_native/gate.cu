// Copyright (c) 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include "gate_kernel.hpp"
__global__ void gate_bsm(const void* a,const void* p,void* out,int m,int n,int k) {
    bf16_bsm_gate_device<maca_bfloat16,maca_bfloat16,float,true>(a,p,out,m,n,k,k,k,m,1.f,0.f,blockIdx.x,blockIdx.y);
}
extern "C" int run_packed_swiglu(void* a,void* p,void* out,int m,void* stream) {
    gate_bsm<<<dim3((m+127)/128,96),256,0,(cudaStream_t)stream>>>(a,p,out,m,12288,2048);
    return int(cudaGetLastError());
}
__global__ void linear_bsm(const void* a,const void* p,void* out,int m,int n,int k) {
    bf16_bsm_gate_device<maca_bfloat16,maca_bfloat16,float,true,false,false>(a,p,out,m,n,k,k,k,m,1.f,0.f,blockIdx.x,blockIdx.y);
}
extern "C" int run_packed_linear(void* a,void* p,void* out,int m,int n,int k,void* stream) {
    linear_bsm<<<dim3((m+127)/128,(n+127)/128),256,0,(cudaStream_t)stream>>>(a,p,out,m,n,k);
    return int(cudaGetLastError());
}
