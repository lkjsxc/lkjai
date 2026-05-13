#include "decoder_cuda_block.hpp"

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__global__ void cached_attention_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out, int total, int layer,
    int start_batch, int position, int cache_batch, int context, int heads,
    int kv_heads, int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % head_dim;
  int h = (i / head_dim) % heads;
  int b = i / (head_dim * heads);
  int kv_h = h / (heads / kv_heads);
  size_t q_base = (static_cast<size_t>(b) * heads + h) * head_dim;
  float scale = rsqrtf(static_cast<float>(head_dim));
  float max_score = -INFINITY;
  for (int t = 0; t <= position; ++t) {
    size_t kb = (((static_cast<size_t>(layer) * cache_batch +
                   (start_batch + b)) *
                      kv_heads +
                  kv_h) *
                     context +
                 t) *
                    head_dim;
    float score = 0.0f;
    for (int x = 0; x < head_dim; ++x) {
      score += __bfloat162float(q[q_base + x]) *
               __bfloat162float(key_cache[kb + x]);
    }
    max_score = fmaxf(max_score, score * scale);
  }
  float denom = 0.0f;
  float value = 0.0f;
  for (int t = 0; t <= position; ++t) {
    size_t vb = (((static_cast<size_t>(layer) * cache_batch +
                   (start_batch + b)) *
                      kv_heads +
                  kv_h) *
                     context +
                 t) *
                    head_dim;
    float score = 0.0f;
    for (int x = 0; x < head_dim; ++x) {
      score += __bfloat162float(q[q_base + x]) *
               __bfloat162float(key_cache[vb + x]);
    }
    float weight = expf(score * scale - max_score);
    denom += weight;
    value += weight * __bfloat162float(value_cache[vb + d]);
  }
  out[i] = __float2bfloat16(value / denom);
}

}  // namespace

void decoder_launch_cached_gqa_attention_bf16(
    const void* q_bf16, const void* key_cache_bf16,
    const void* value_cache_bf16, void* out_bf16, int layer, int start_batch,
    int position, int cache_batch, int context, int batch, int heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  if (batch <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0) return;
  if (heads % kv_heads != 0 || position < 0 || position >= context) {
    throw std::runtime_error("decoder cached GQA attention shape mismatch");
  }
  int total = batch * heads * head_dim;
  cached_attention_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_bf16),
      static_cast<const __nv_bfloat16*>(key_cache_bf16),
      static_cast<const __nv_bfloat16*>(value_cache_bf16),
      static_cast<__nv_bfloat16*>(out_bf16), total, layer, start_batch,
      position, cache_batch, context, heads, kv_heads, head_dim);
  require_cuda(cudaGetLastError(), "decoder_cached_gqa_attention_bf16_kernel");
}

}  // namespace lkjai
