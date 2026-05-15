#include "decoder_cuda_block.hpp"

#include <cstddef>
#include <stdexcept>

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void rope_backward_bf16_kernel(const __nv_bfloat16* d_output,
                                          __nv_bfloat16* d_input,
                                          int total_pairs, int seq, int heads,
                                          int head_dim, int position_offset,
                                          float theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_pairs) return;
  int pair = i % (head_dim / 2);
  int row = i / (head_dim / 2);
  int pos = position_offset + (row / heads) % seq;
  auto* dst = d_input + static_cast<size_t>(row) * head_dim + pair * 2;
  const auto* src = d_output + static_cast<size_t>(row) * head_dim + pair * 2;
  float g0 = __bfloat162float(src[0]);
  float g1 = __bfloat162float(src[1]);
  float inv = powf(theta, -2.0f * static_cast<float>(pair) /
                             static_cast<float>(head_dim));
  float angle = static_cast<float>(pos) * inv;
  float c = cosf(angle);
  float s = sinf(angle);
  dst[0] = __float2bfloat16(g0 * c + g1 * s);
  dst[1] = __float2bfloat16(-g0 * s + g1 * c);
}

}  // namespace

void decoder_launch_rope_backward_bf16_at(const void* d_output_bf16,
                                          void* d_input_bf16, int batch,
                                          int seq, int heads, int head_dim,
                                          int position_offset, float theta,
                                          cudaStream_t stream) {
  if (batch <= 0 || seq <= 0 || heads <= 0 || head_dim <= 0) return;
  if (head_dim % 2 != 0) {
    throw std::runtime_error("decoder RoPE backward requires even head_dim");
  }
  int total_pairs = batch * seq * heads * (head_dim / 2);
  rope_backward_bf16_kernel<<<(total_pairs + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(d_output_bf16),
      static_cast<__nv_bfloat16*>(d_input_bf16), total_pairs, seq, heads,
      head_dim, position_offset, theta);
  require_cuda(cudaGetLastError(), "decoder_rope_backward_bf16_kernel");
}

}  // namespace lkjai
