#pragma once

#include <cstdint>
#include <string>

namespace lkjai {

struct DecoderKvCacheConfig {
  int layers = 0;
  int batch = 0;
  int kv_heads = 0;
  int context = 0;
  int head_dim = 0;
};

struct DecoderKvCacheLayout {
  DecoderKvCacheConfig cfg;
  uint64_t values_per_tensor = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t total_bytes = 0;
};

bool decoder_kv_cache_layout(const DecoderKvCacheConfig& cfg,
                             DecoderKvCacheLayout* layout,
                             std::string* error);
uint64_t decoder_kv_cache_value_offset(const DecoderKvCacheLayout& layout,
                                       int layer, int batch, int kv_head,
                                       int position, int dim);
uint64_t decoder_kv_cache_byte_offset(const DecoderKvCacheLayout& layout,
                                      bool value_tensor, int layer, int batch,
                                      int kv_head, int position, int dim);

}  // namespace lkjai
