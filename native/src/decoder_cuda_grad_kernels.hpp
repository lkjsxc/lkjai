#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace lkjai {

void decoder_cuda_zero_f32(float* data, int elements, cudaStream_t stream);
void decoder_cuda_add_lm_head_grad(const float* grad_logits,
                                   const void* hidden_bf16, float* grad,
                                   int rows, int vocab, int hidden,
                                   cudaStream_t stream);
void decoder_cuda_add_first_embedding_grad(const uint16_t* tokens,
                                           const void* hidden_row_bf16,
                                           float* grad, int vocab, int hidden,
                                           float scale, cudaStream_t stream);
void decoder_cuda_add_signed_hidden_grad(const void* hidden_row_bf16,
                                         float* grad, int elements,
                                         int hidden, float scale,
                                         cudaStream_t stream);
void decoder_cuda_add_constant_grad(float* grad, int elements, float scale,
                                    cudaStream_t stream);

}  // namespace lkjai
