#include "decoder_kv_cache.hpp"

#include <utility>

#include <cuda_runtime.h>

namespace lkjai {
namespace {

void free_cache(DecoderKvCache* cache) {
  if (cache->key_device) cudaFree(cache->key_device);
  if (cache->value_device) cudaFree(cache->value_device);
  cache->key_device = nullptr;
  cache->value_device = nullptr;
  cache->allocated_bytes = 0;
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

}  // namespace lkjai
