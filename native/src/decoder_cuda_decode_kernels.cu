#include "decoder_cuda_decode_kernels.hpp"

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void gather_embeddings_kernel(const __nv_bfloat16* table,
                                         const uint16_t* tokens,
                                         __nv_bfloat16* out, int rows,
                                         int hidden, int vocab) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * hidden;
  if (i >= total) return;
  int row = i / hidden;
  int col = i % hidden;
  int token = static_cast<int>(tokens[row]) % vocab;
  out[i] = table[static_cast<size_t>(token) * hidden + col];
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* in, float* out,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

}  // namespace

void decoder_cuda_gather_embeddings_bf16(const void* table_bf16,
                                         const void* tokens_u16, void* out_bf16,
                                         int rows, int hidden, int vocab,
                                         cudaStream_t stream) {
  int total = rows * hidden;
  gather_embeddings_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table_bf16),
      static_cast<const uint16_t*>(tokens_u16),
      static_cast<__nv_bfloat16*>(out_bf16), rows, hidden, vocab);
  require_cuda(cudaGetLastError(), "decoder gather embeddings");
}

void decoder_cuda_bf16_to_f32(const void* in_bf16, void* out_f32, int elements,
                              cudaStream_t stream) {
  bf16_to_f32_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(in_bf16),
      static_cast<float*>(out_f32), elements);
  require_cuda(cudaGetLastError(), "decoder bf16 to f32");
}

}  // namespace lkjai
