#include "decoder_cuda_norm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__device__ float block_sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* input,
                                    const float* weight,
                                    __nv_bfloat16* output, int rows,
                                    int hidden, float eps) {
  extern __shared__ float scratch[];
  int row = blockIdx.x;
  if (row >= rows) return;
  const auto* x = input + static_cast<size_t>(row) * hidden;
  auto* y = output + static_cast<size_t>(row) * hidden;
  float local = 0.0f;
  for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
    float v = __bfloat162float(x[h]);
    local += v * v;
  }
  float ss = block_sum(local, scratch);
  float scale = rsqrtf(ss / static_cast<float>(hidden) + eps);
  for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
    float v = __bfloat162float(x[h]) * scale * weight[h];
    y[h] = __float2bfloat16(v);
  }
}

__global__ void rmsnorm_backward_bf16_kernel(
    const __nv_bfloat16* input, const float* weight,
    const __nv_bfloat16* d_output, float* d_input, float* d_weight, int rows,
    int hidden, float eps) {
  extern __shared__ float scratch[];
  int row = blockIdx.x;
  if (row >= rows) return;
  const auto* x = input + static_cast<size_t>(row) * hidden;
  const auto* dy = d_output + static_cast<size_t>(row) * hidden;
  auto* dx = d_input + static_cast<size_t>(row) * hidden;
  float local_ss = 0.0f;
  float local_dot = 0.0f;
  for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
    float xv = __bfloat162float(x[h]);
    float dyv = __bfloat162float(dy[h]);
    local_ss += xv * xv;
    local_dot += dyv * weight[h] * xv;
  }
  float ss = block_sum(local_ss, scratch);
  float dot = block_sum(local_dot, scratch);
  float inv = rsqrtf(ss / static_cast<float>(hidden) + eps);
  float coeff = inv * inv * inv * dot / static_cast<float>(hidden);
  for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
    float xv = __bfloat162float(x[h]);
    float dyv = __bfloat162float(dy[h]);
    dx[h] = dyv * weight[h] * inv - xv * coeff;
    atomicAdd(d_weight + h, dyv * xv * inv);
  }
}

}  // namespace

void decoder_launch_rmsnorm_bf16(const void* input_bf16,
                                 const float* weight_f32,
                                 void* output_bf16, int rows, int hidden,
                                 float eps, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  rmsnorm_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(input_bf16), weight_f32,
      static_cast<__nv_bfloat16*>(output_bf16), rows, hidden, eps);
  require_cuda(cudaGetLastError(), "decoder_rmsnorm_bf16_kernel");
}

void decoder_launch_rmsnorm_backward_bf16(
    const void* input_bf16, const float* weight_f32, const void* d_output_bf16,
    float* d_input_f32, float* d_weight_f32, int rows, int hidden, float eps,
    float d_weight_beta, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  if (d_weight_beta == 0.0f) {
    require_cuda(cudaMemsetAsync(d_weight_f32, 0,
                                 static_cast<size_t>(hidden) * sizeof(float),
                                 stream),
                 "decoder rmsnorm d_weight zero");
  } else if (d_weight_beta != 1.0f) {
    throw std::runtime_error("decoder RMSNorm backward supports beta 0 or 1");
  }
  rmsnorm_backward_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(input_bf16), weight_f32,
      static_cast<const __nv_bfloat16*>(d_output_bf16), d_input_f32,
      d_weight_f32, rows, hidden, eps);
  require_cuda(cudaGetLastError(), "decoder_rmsnorm_backward_bf16_kernel");
}

}  // namespace lkjai
