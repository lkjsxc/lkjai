#include "decoder_cuda_norm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

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

}  // namespace lkjai
