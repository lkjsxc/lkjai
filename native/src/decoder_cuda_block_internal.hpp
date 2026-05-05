#pragma once

#include <cstddef>

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace lkjai {

void decoder_cuda_project_bf16(cublasLtHandle_t handle, cudaStream_t stream,
                               const void* x_bf16, const void* w_bf16,
                               void* y_bf16, int rows, int in_features,
                               int out_features, void* workspace,
                               size_t workspace_bytes);
size_t decoder_cuda_projection_plan_cache_size();
void decoder_cuda_projection_plan_cache_reset();

}  // namespace lkjai
