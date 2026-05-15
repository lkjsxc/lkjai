#pragma once

#include <cuda_runtime.h>

namespace lkjai {

void decoder_launch_rmsnorm_bf16(const void* input_bf16,
                                 const float* weight_f32,
                                 void* output_bf16, int rows, int hidden,
                                 float eps, cudaStream_t stream);
void decoder_launch_rmsnorm_backward_bf16(
    const void* input_bf16, const float* weight_f32, const void* d_output_bf16,
    float* d_input_f32, float* d_weight_f32, int rows, int hidden, float eps,
    float d_weight_beta, cudaStream_t stream);

}  // namespace lkjai
