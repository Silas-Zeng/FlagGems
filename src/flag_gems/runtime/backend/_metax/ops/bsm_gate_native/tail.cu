// Copyright (c) 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include "gate_kernel.hpp"

// For M1921..2048 the 128x128 grid contains 256 tiles. On 104 SMs,
// 208 full tiles fill two waves; split the final48 tiles into two K ranges.
// Only those final tiles use FP32 temporary storage and a separate reduction.
__global__ void down_tail_bsm(const void* a, const void* packed, void* output,
                             float* partials, int m) {
    constexpr int n = 2048, k = 6144, prefix = 208, tail = 48;
    const int id = blockIdx.x;
    const bool partial = id >= prefix;
    const int split = partial ? (id - prefix) / tail : 0;
    const int tile = partial ? prefix + (id - prefix) % tail : id;
    const int ks = partial ? k / 2 : k;
    void* destination = partial
        ? static_cast<void*>(partials + size_t(split) * m * n) : output;
    bf16_bsm_gate_device<maca_bfloat16, maca_bfloat16, float, true, false, false>(
        static_cast<const maca_bfloat16*>(a) + split * ks,
        static_cast<const maca_bfloat16*>(packed) + split * (ks / 64) * 8192,
        destination, m, n, ks, k, k, n, 1.f, 0.f, tile % 16, tile / 16, partial);
}

__global__ void reduce_down_tail(const float* partials, maca_bfloat16* output,
                                 int m) {
    constexpr int n = 2048, prefix = 208, tail = 48;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= tail * 128 * 128) return;
    const int tile = prefix + i / (128 * 128);
    const int row = (tile % 16) * 128 + (i % (128 * 128)) / 128;
    const int col = (tile / 16) * 128 + i % 128;
    if (row < m) {
        const size_t offset = size_t(row) * n + col;
        const float value = partials[offset] + partials[size_t(m) * n + offset];
        output[offset] = maca_bfloat16(value);
    }
}

extern "C" int run_bsm_tail(void* a, void* packed, void* output, void* partials,
                            int m, int n, int k, int splits, int sm_count,
                            void* stream) {
    if (m < 1921 || m > 2048 || n != 2048 || k != 6144
        || splits != 2 || sm_count != 104) return int(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    down_tail_bsm<<<304, 256, 0, s>>>(a, packed, output, static_cast<float*>(partials), m);
    reduce_down_tail<<<3072, 256, 0, s>>>(static_cast<float*>(partials),
                                        static_cast<maca_bfloat16*>(output), m);
    return int(cudaGetLastError());
}
