#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

struct DecoderKvCache {
  DecoderKvCacheLayout layout;
  std::vector<uint16_t> key;
  std::vector<uint16_t> value;
  std::vector<int> next_position;
  uint64_t allocated_bytes = 0;
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
bool decoder_kv_cache_allocate(const DecoderKvCacheConfig& cfg,
                               DecoderKvCache* cache, std::string* error);
bool decoder_kv_cache_write(DecoderKvCache* cache, bool value_tensor,
                            int layer, int batch, int kv_head, int position,
                            int dim, uint16_t value, std::string* error);
bool decoder_kv_cache_read(const DecoderKvCache& cache, bool value_tensor,
                           int layer, int batch, int kv_head, int position,
                           int dim, uint16_t* value, std::string* error);
bool decoder_kv_cache_append(DecoderKvCache* cache, int batch,
                             const std::vector<uint16_t>& key_values,
                             const std::vector<uint16_t>& value_values,
                             std::string* error);

}  // namespace lkjai
