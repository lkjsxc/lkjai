#include "decoder_kv_cache.hpp"

#include <utility>

#include <cuda_runtime.h>

namespace lkjai {
namespace {

bool in_bounds(const DecoderKvCacheLayout& layout, int layer, int batch,
               int kv_head, int position, int dim, std::string* error) {
  const auto& c = layout.cfg;
  bool ok = layer >= 0 && layer < c.layers && batch >= 0 && batch < c.batch &&
            kv_head >= 0 && kv_head < c.kv_heads && position >= 0 &&
            position < c.context && dim >= 0 && dim < c.head_dim;
  if (!ok) *error = "decoder KV cache index out of bounds";
  return ok;
}

void free_cache(DecoderKvCache* cache) {
  if (cache->key_device) cudaFree(cache->key_device);
  if (cache->value_device) cudaFree(cache->value_device);
  cache->key_device = nullptr;
  cache->value_device = nullptr;
  cache->allocated_bytes = 0;
}

bool cuda_ok(cudaError_t status, const char* label, std::string* error) {
  if (status == cudaSuccess) return true;
  *error = std::string(label) + ": " + cudaGetErrorString(status);
  return false;
}

}  // namespace

DecoderKvCache::DecoderKvCache(DecoderKvCache&& other) noexcept {
  *this = std::move(other);
}

DecoderKvCache& DecoderKvCache::operator=(DecoderKvCache&& other) noexcept {
  if (this == &other) return *this;
  free_cache(this);
  layout = other.layout;
  key_device = other.key_device;
  value_device = other.value_device;
  next_position = std::move(other.next_position);
  allocated_bytes = other.allocated_bytes;
  other.key_device = nullptr;
  other.value_device = nullptr;
  other.allocated_bytes = 0;
  return *this;
}

DecoderKvCache::~DecoderKvCache() { free_cache(this); }

bool decoder_kv_cache_layout(const DecoderKvCacheConfig& cfg,
                             DecoderKvCacheLayout* layout,
                             std::string* error) {
  if (cfg.layers <= 0 || cfg.batch <= 0 || cfg.kv_heads <= 0 ||
      cfg.context <= 0 || cfg.head_dim <= 0) {
    *error = "decoder KV cache dimensions must be positive";
    return false;
  }
  if (cfg.head_dim % 8 != 0) {
    *error = "decoder KV cache head_dim must be a multiple of 8";
    return false;
  }
  DecoderKvCacheLayout out;
  out.cfg = cfg;
  out.values_per_tensor = static_cast<uint64_t>(cfg.layers) * cfg.batch *
                          cfg.kv_heads * cfg.context * cfg.head_dim;
  out.bytes_per_tensor = out.values_per_tensor * 2u;
  out.total_bytes = out.bytes_per_tensor * 2u;
  *layout = out;
  return true;
}

uint64_t decoder_kv_cache_value_offset(const DecoderKvCacheLayout& layout,
                                       int layer, int batch, int kv_head,
                                       int position, int dim) {
  const auto& c = layout.cfg;
  uint64_t offset = static_cast<uint64_t>(layer);
  offset = offset * c.batch + static_cast<uint64_t>(batch);
  offset = offset * c.kv_heads + static_cast<uint64_t>(kv_head);
  offset = offset * c.context + static_cast<uint64_t>(position);
  offset = offset * c.head_dim + static_cast<uint64_t>(dim);
  return offset;
}

uint64_t decoder_kv_cache_byte_offset(const DecoderKvCacheLayout& layout,
                                      bool value_tensor, int layer, int batch,
                                      int kv_head, int position, int dim) {
  uint64_t offset = decoder_kv_cache_value_offset(
      layout, layer, batch, kv_head, position, dim) * 2u;
  return value_tensor ? layout.bytes_per_tensor + offset : offset;
}

bool decoder_kv_cache_allocate(const DecoderKvCacheConfig& cfg,
                               DecoderKvCache* cache, std::string* error) {
  DecoderKvCacheLayout layout;
  if (!decoder_kv_cache_layout(cfg, &layout, error)) return false;
  free_cache(cache);
  cache->layout = layout;
  cache->next_position.assign(static_cast<size_t>(cfg.batch), 0);
  if (!cuda_ok(cudaMalloc(&cache->key_device, layout.bytes_per_tensor),
               "decoder KV cudaMalloc key", error)) {
    free_cache(cache);
    return false;
  }
  if (!cuda_ok(cudaMalloc(&cache->value_device, layout.bytes_per_tensor),
               "decoder KV cudaMalloc value", error)) {
    free_cache(cache);
    return false;
  }
  if (!cuda_ok(cudaMemset(cache->key_device, 0, layout.bytes_per_tensor),
               "decoder KV cudaMemset key", error) ||
      !cuda_ok(cudaMemset(cache->value_device, 0, layout.bytes_per_tensor),
               "decoder KV cudaMemset value", error)) {
    free_cache(cache);
    return false;
  }
  cache->allocated_bytes = layout.total_bytes;
  return true;
}

bool decoder_kv_cache_write(DecoderKvCache* cache, bool value_tensor,
                            int layer, int batch, int kv_head, int position,
                            int dim, uint16_t value, std::string* error) {
  if (!in_bounds(cache->layout, layer, batch, kv_head, position, dim, error)) {
    return false;
  }
  auto offset = static_cast<size_t>(decoder_kv_cache_value_offset(
      cache->layout, layer, batch, kv_head, position, dim));
  auto* dst = static_cast<uint16_t*>(
      value_tensor ? cache->value_device : cache->key_device);
  return cuda_ok(cudaMemcpy(dst + offset, &value, sizeof(uint16_t),
                            cudaMemcpyHostToDevice),
                 "decoder KV write", error);
}

bool decoder_kv_cache_read(const DecoderKvCache& cache, bool value_tensor,
                           int layer, int batch, int kv_head, int position,
                           int dim, uint16_t* value, std::string* error) {
  if (!in_bounds(cache.layout, layer, batch, kv_head, position, dim, error)) {
    return false;
  }
  auto offset = static_cast<size_t>(decoder_kv_cache_value_offset(
      cache.layout, layer, batch, kv_head, position, dim));
  auto* src = static_cast<const uint16_t*>(
      value_tensor ? cache.value_device : cache.key_device);
  return cuda_ok(cudaMemcpy(value, src + offset, sizeof(uint16_t),
                            cudaMemcpyDeviceToHost),
                 "decoder KV read", error);
}

bool decoder_kv_cache_append(DecoderKvCache* cache, int batch,
                             const std::vector<uint16_t>& key_values,
                             const std::vector<uint16_t>& value_values,
                             std::string* error) {
  const auto& c = cache->layout.cfg;
  if (batch < 0 || batch >= c.batch ||
      cache->next_position[static_cast<size_t>(batch)] >= c.context) {
    *error = "decoder KV cache append position out of bounds";
    return false;
  }
  size_t per_token = static_cast<size_t>(c.layers * c.kv_heads * c.head_dim);
  if (key_values.size() != per_token || value_values.size() != per_token) {
    *error = "decoder KV cache append vector size mismatch";
    return false;
  }
  int position = cache->next_position[static_cast<size_t>(batch)]++;
  size_t i = 0;
  for (int layer = 0; layer < c.layers; ++layer)
    for (int head = 0; head < c.kv_heads; ++head)
      for (int dim = 0; dim < c.head_dim; ++dim, ++i) {
        decoder_kv_cache_write(cache, false, layer, batch, head, position, dim,
                               key_values[i], error);
        decoder_kv_cache_write(cache, true, layer, batch, head, position, dim,
                               value_values[i], error);
      }
  return true;
}

}  // namespace lkjai
