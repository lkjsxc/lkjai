#include "decoder_kv_cache.hpp"

#include <utility>

#include <cuda_runtime.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

void free_cache(DecoderKvCache* cache) {
  if (cache->key_device) {
    cudaFree(cache->key_device);
    device_allocation_account_free(cache->layout.bytes_per_tensor);
  }
  if (cache->value_device) {
    cudaFree(cache->value_device);
    device_allocation_account_free(cache->layout.bytes_per_tensor);
  }
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
