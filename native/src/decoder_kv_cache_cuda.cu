#include "decoder_kv_cache.hpp"

#include <cuda_bf16.h>

namespace lkjai {
namespace {

bool cuda_ok(cudaError_t status, const char* label, std::string* error) {
  if (status == cudaSuccess) return true;
  *error = std::string(label) + ": " + cudaGetErrorString(status);
  return false;
}

__global__ void append_layer_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* key_cache, __nv_bfloat16* value_cache, int total, int layer,
    int start_position, int cache_batch, int context, int seq, int kv_heads,
    int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % head_dim;
  int h = (i / head_dim) % kv_heads;
  int t = (i / (head_dim * kv_heads)) % seq;
  int b = i / (head_dim * kv_heads * seq);
  size_t src = ((static_cast<size_t>(b) * seq + t) * kv_heads + h) * head_dim + d;
  size_t dst = (((static_cast<size_t>(layer) * cache_batch + b) * kv_heads + h) *
                    context +
                (start_position + t)) *
                   head_dim +
               d;
  key_cache[dst] = key[src];
  value_cache[dst] = value[src];
}

}  // namespace

bool decoder_kv_cache_append_device_layer(DecoderKvCache* cache, int layer,
                                          int start_position,
                                          const void* key_bf16,
                                          const void* value_bf16, int batch,
                                          int seq, cudaStream_t stream,
                                          std::string* error) {
  const auto& c = cache->layout.cfg;
  if (layer < 0 || layer >= c.layers || batch <= 0 || batch > c.batch ||
      start_position < 0 || start_position + seq > c.context) {
    *error = "decoder device KV append shape out of bounds";
    return false;
  }
  int total = batch * seq * c.kv_heads * c.head_dim;
  append_layer_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(key_bf16),
      static_cast<const __nv_bfloat16*>(value_bf16),
      static_cast<__nv_bfloat16*>(cache->key_device),
      static_cast<__nv_bfloat16*>(cache->value_device), total, layer,
      start_position, c.batch, c.context, seq, c.kv_heads, c.head_dim);
  if (!cuda_ok(cudaGetLastError(), "decoder device KV append", error)) {
    return false;
  }
  if (layer == c.layers - 1) {
    for (int b = 0; b < batch; ++b) cache->next_position[b] = start_position + seq;
  }
  return true;
}

}  // namespace lkjai
