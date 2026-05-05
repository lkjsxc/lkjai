#pragma once

#include <cuda_runtime.h>

namespace lkjai {

void decoder_launch_residual_add_bf16(const void* lhs_bf16,
                                      const void* rhs_bf16,
                                      void* out_bf16, int elements,
                                      cudaStream_t stream);

}  // namespace lkjai
