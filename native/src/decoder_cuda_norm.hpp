#pragma once

#include <cuda_runtime.h>

namespace lkjai {

void decoder_launch_rmsnorm_bf16(const void* input_bf16,
                                 const float* weight_f32,
                                 void* output_bf16, int rows, int hidden,
                                 float eps, cudaStream_t stream);

}  // namespace lkjai
