#include "decoder_cuda_residual.hpp"

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void residual_add_bf16_kernel(const __nv_bfloat16* lhs,
                                         const __nv_bfloat16* rhs,
                                         __nv_bfloat16* out, int elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  float value = __bfloat162float(lhs[i]) + __bfloat162float(rhs[i]);
  out[i] = __float2bfloat16(value);
}

}  // namespace

void decoder_launch_residual_add_bf16(const void* lhs_bf16,
                                      const void* rhs_bf16,
                                      void* out_bf16, int elements,
                                      cudaStream_t stream) {
  if (elements <= 0) return;
  residual_add_bf16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(lhs_bf16),
      static_cast<const __nv_bfloat16*>(rhs_bf16),
      static_cast<__nv_bfloat16*>(out_bf16), elements);
  require_cuda(cudaGetLastError(), "decoder_residual_add_bf16_kernel");
}

}  // namespace lkjai
