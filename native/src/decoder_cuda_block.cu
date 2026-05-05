#include "decoder_cuda_block.hpp"

#include <cstddef>
#include <stdexcept>

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void rope_bf16_kernel(__nv_bfloat16* tensor, int total_pairs,
                                 int seq, int heads, int head_dim,
                                 float theta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_pairs) return;
  int pair = i % (head_dim / 2);
  int row = i / (head_dim / 2);
  int pos = (row / heads) % seq;
  auto* base = tensor + static_cast<size_t>(row) * head_dim + pair * 2;
  float x0 = __bfloat162float(base[0]);
  float x1 = __bfloat162float(base[1]);
  float inv = powf(theta, -2.0f * static_cast<float>(pair) /
                             static_cast<float>(head_dim));
  float angle = static_cast<float>(pos) * inv;
  float c = cosf(angle);
  float s = sinf(angle);
  base[0] = __float2bfloat16(x0 * c - x1 * s);
  base[1] = __float2bfloat16(x0 * s + x1 * c);
}

__global__ void swiglu_bf16_kernel(const __nv_bfloat16* gate,
                                   const __nv_bfloat16* up,
                                   __nv_bfloat16* out, int elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  float g = __bfloat162float(gate[i]);
  float u = __bfloat162float(up[i]);
  out[i] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

__global__ void causal_gqa_attention_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, int total, int seq, int heads, int kv_heads,
    int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % head_dim;
  int h = (i / head_dim) % heads;
  int t = (i / (head_dim * heads)) % seq;
  int b = i / (head_dim * heads * seq);
  int kv_h = h % kv_heads;
  size_t q_base =
      ((static_cast<size_t>(b) * seq + t) * heads + h) * head_dim;
  float scale = rsqrtf(static_cast<float>(head_dim));
  float max_score = -INFINITY;
  for (int s = 0; s <= t; ++s) {
    size_t k_base =
        ((static_cast<size_t>(b) * seq + s) * kv_heads + kv_h) * head_dim;
    float score = 0.0f;
    for (int x = 0; x < head_dim; ++x) {
      score += __bfloat162float(q[q_base + x]) *
               __bfloat162float(k[k_base + x]);
    }
    max_score = fmaxf(max_score, score * scale);
  }
  float denom = 0.0f;
  float value = 0.0f;
  for (int s = 0; s <= t; ++s) {
    size_t k_base =
        ((static_cast<size_t>(b) * seq + s) * kv_heads + kv_h) * head_dim;
    size_t v_base = k_base;
    float score = 0.0f;
    for (int x = 0; x < head_dim; ++x) {
      score += __bfloat162float(q[q_base + x]) *
               __bfloat162float(k[k_base + x]);
    }
    float weight = expf(score * scale - max_score);
    denom += weight;
    value += weight * __bfloat162float(v[v_base + d]);
  }
  out[i] = __float2bfloat16(value / denom);
}

}  // namespace

void decoder_launch_rope_bf16(void* tensor_bf16, int batch, int seq, int heads,
                              int head_dim, float theta,
                              cudaStream_t stream) {
  if (batch <= 0 || seq <= 0 || heads <= 0 || head_dim <= 0) return;
  if (head_dim % 2 != 0) {
    throw std::runtime_error("decoder RoPE requires even head_dim");
  }
  int total_pairs = batch * seq * heads * (head_dim / 2);
  rope_bf16_kernel<<<(total_pairs + 255) / 256, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(tensor_bf16), total_pairs, seq, heads,
      head_dim, theta);
  require_cuda(cudaGetLastError(), "decoder_rope_bf16_kernel");
}

void decoder_launch_swiglu_bf16(const void* gate_bf16, const void* up_bf16,
                                void* out_bf16, int elements,
                                cudaStream_t stream) {
  if (elements <= 0) return;
  swiglu_bf16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_bf16),
      static_cast<const __nv_bfloat16*>(up_bf16),
      static_cast<__nv_bfloat16*>(out_bf16), elements);
  require_cuda(cudaGetLastError(), "decoder_swiglu_bf16_kernel");
}

void decoder_launch_causal_gqa_attention_bf16(
    const void* q_bf16, const void* k_bf16, const void* v_bf16, void* out_bf16,
    int batch, int seq, int heads, int kv_heads, int head_dim,
    cudaStream_t stream) {
  if (batch <= 0 || seq <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  if (heads % kv_heads != 0) {
    throw std::runtime_error("decoder GQA attention requires heads % kv_heads == 0");
  }
  int total = batch * seq * heads * head_dim;
  causal_gqa_attention_bf16_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_bf16),
      static_cast<const __nv_bfloat16*>(k_bf16),
      static_cast<const __nv_bfloat16*>(v_bf16),
      static_cast<__nv_bfloat16*>(out_bf16), total, seq, heads, kv_heads,
      head_dim);
  require_cuda(cudaGetLastError(), "decoder_causal_gqa_attention_bf16_kernel");
}

}  // namespace lkjai
