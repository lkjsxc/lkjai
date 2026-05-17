#include "decoder_cuda_grad_kernels.hpp"

#include <cstdint>

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void zero_kernel(float* data, int elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elements) data[i] = 0.0f;
}

__global__ void lm_head_grad_kernel(const float* grad_logits,
                                    const __nv_bfloat16* hidden, float* grad,
                                    int rows, int vocab, int hidden_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = vocab * hidden_size;
  if (i >= total) return;
  int h = i % hidden_size;
  int v = i / hidden_size;
  float sum = 0.0f;
  for (int r = 0; r < rows; ++r) {
    sum += grad_logits[static_cast<size_t>(r) * vocab + v] *
           __bfloat162float(hidden[static_cast<size_t>(r) * hidden_size + h]);
  }
  atomicAdd(grad + i, sum);
}

__global__ void first_embedding_grad_kernel(
    const uint16_t* tokens, const __nv_bfloat16* hidden, float* grad,
    int vocab, int hidden_size, float scale) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden_size) return;
  int token = static_cast<int>(tokens[0]) % vocab;
  atomicAdd(grad + static_cast<size_t>(token) * hidden_size + h,
            scale * __bfloat162float(hidden[h]));
}

__global__ void signed_hidden_grad_kernel(const __nv_bfloat16* hidden,
                                          float* grad, int elements,
                                          int hidden_size, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  float h = __bfloat162float(hidden[i % hidden_size]);
  grad[i] += scale * (h >= 0.0f ? 1.0f : -1.0f);
}

__global__ void constant_grad_kernel(float* grad, int elements, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elements) grad[i] += scale;
}

int blocks(int elements) { return (elements + 255) / 256; }

}  // namespace

void decoder_cuda_zero_f32(float* data, int elements, cudaStream_t stream) {
  if (elements <= 0) return;
  zero_kernel<<<blocks(elements), 256, 0, stream>>>(data, elements);
  require_cuda(cudaGetLastError(), "decoder zero f32 grad");
}

void decoder_cuda_add_lm_head_grad(const float* grad_logits,
                                   const void* hidden_bf16, float* grad,
                                   int rows, int vocab, int hidden,
                                   cudaStream_t stream) {
  int elements = vocab * hidden;
  if (rows <= 0 || elements <= 0) return;
  lm_head_grad_kernel<<<blocks(elements), 256, 0, stream>>>(
      grad_logits, static_cast<const __nv_bfloat16*>(hidden_bf16), grad, rows,
      vocab, hidden);
  require_cuda(cudaGetLastError(), "decoder LM-head grad");
}

void decoder_cuda_add_first_embedding_grad(const uint16_t* tokens,
                                           const void* hidden_row_bf16,
                                           float* grad, int vocab, int hidden,
                                           float scale, cudaStream_t stream) {
  if (hidden <= 0) return;
  first_embedding_grad_kernel<<<blocks(hidden), 256, 0, stream>>>(
      tokens, static_cast<const __nv_bfloat16*>(hidden_row_bf16), grad, vocab,
      hidden, scale);
  require_cuda(cudaGetLastError(), "decoder embedding grad");
}

void decoder_cuda_add_signed_hidden_grad(const void* hidden_row_bf16,
                                         float* grad, int elements,
                                         int hidden, float scale,
                                         cudaStream_t stream) {
  if (elements <= 0) return;
  signed_hidden_grad_kernel<<<blocks(elements), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(hidden_row_bf16), grad, elements,
      hidden, scale);
  require_cuda(cudaGetLastError(), "decoder signed hidden grad");
}

void decoder_cuda_add_constant_grad(float* grad, int elements, float scale,
                                    cudaStream_t stream) {
  if (elements <= 0) return;
  constant_grad_kernel<<<blocks(elements), 256, 0, stream>>>(grad, elements,
                                                            scale);
  require_cuda(cudaGetLastError(), "decoder constant grad");
}

}  // namespace lkjai
