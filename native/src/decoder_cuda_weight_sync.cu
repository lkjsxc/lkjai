#include "decoder_cuda_weight_sync.hpp"

#include <cstdint>

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void copy_f32_kernel(const float* src, float* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

__global__ void transpose_bf16_kernel(const __nv_bfloat16* src,
                                      __nv_bfloat16* dst, int rows,
                                      int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * cols;
  if (i >= n) return;
  int row = i / cols;
  int col = i % cols;
  dst[col * rows + row] = src[i];
}

}  // namespace

void decoder_cuda_copy_f32_device(const void* src, void* dst, int elements,
                                  cudaStream_t stream) {
  copy_f32_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const float*>(src), static_cast<float*>(dst), elements);
  require_cuda(cudaGetLastError(), "decoder copy f32 device");
}

void decoder_cuda_transpose_bf16_device(const void* src, void* dst, int rows,
                                        int cols, cudaStream_t stream) {
  int n = rows * cols;
  transpose_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(src), static_cast<__nv_bfloat16*>(dst),
      rows, cols);
  require_cuda(cudaGetLastError(), "decoder transpose bf16 device");
}

}  // namespace lkjai
