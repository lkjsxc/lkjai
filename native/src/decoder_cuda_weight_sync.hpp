#pragma once

#include <cuda_runtime.h>

namespace lkjai {

void decoder_cuda_copy_f32_device(const void* src, void* dst, int elements,
                                  cudaStream_t stream);
void decoder_cuda_transpose_bf16_device(const void* src, void* dst, int rows,
                                        int cols, cudaStream_t stream);

}  // namespace lkjai
