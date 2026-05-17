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
void decoder_cuda_project_backward_bf16(
    cublasLtHandle_t handle, cudaStream_t stream, const void* x_bf16,
    const void* w_bf16, const void* dy_bf16, void* dx_f32, void* dw_f32,
    int rows, int in_features, int out_features, void* workspace,
    size_t workspace_bytes, float dw_beta);
void decoder_cuda_project_backward_param_layout_bf16(
    cublasLtHandle_t handle, cudaStream_t stream, const void* x_bf16,
    const void* w_forward_bf16, const void* dy_bf16, void* dx_f32,
    void* dw_param_f32, int rows, int in_features, int out_features,
    void* workspace, size_t workspace_bytes, float dw_beta);
void decoder_cuda_lm_head_dhidden_f32(
    cublasLtHandle_t handle, cudaStream_t stream, const void* grad_logits_f32,
    const void* lm_head_f32, void* d_hidden_f32, int rows, int vocab,
    int hidden, void* workspace, size_t workspace_bytes);
size_t decoder_cuda_projection_plan_cache_size();
void decoder_cuda_projection_plan_cache_reset();

}  // namespace lkjai
