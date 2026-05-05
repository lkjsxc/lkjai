#include "decoder_kv_cache.hpp"

namespace lkjai {

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

}  // namespace lkjai
