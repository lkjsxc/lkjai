#include <iostream>
#include <string>

#include "decoder_kv_cache.hpp"

namespace {

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

}  // namespace

int main() {
  lkjai::DecoderKvCacheLayout layout;
  std::string error;
  lkjai::DecoderKvCacheConfig cfg{10, 2, 2, 1024, 72};
  if (!expect(lkjai::decoder_kv_cache_layout(cfg, &layout, &error), error)) {
    return 1;
  }
  uint64_t values = 10ull * 2ull * 2ull * 1024ull * 72ull;
  bool ok = expect(layout.values_per_tensor == values, "value count") &&
            expect(layout.bytes_per_tensor == values * 2ull, "bytes per K/V") &&
            expect(layout.total_bytes == values * 4ull, "total bytes");
  ok = ok && expect(decoder_kv_cache_value_offset(layout, 1, 0, 0, 0, 0) ==
                        2ull * 2ull * 1024ull * 72ull,
                    "layer offset");
  ok = ok && expect(decoder_kv_cache_value_offset(layout, 0, 1, 0, 0, 0) ==
                        2ull * 1024ull * 72ull,
                    "batch offset");
  ok = ok && expect(decoder_kv_cache_value_offset(layout, 0, 0, 1, 0, 0) ==
                        1024ull * 72ull,
                    "kv head offset");
  ok = ok && expect(decoder_kv_cache_byte_offset(layout, true, 0, 0, 0, 0, 0) ==
                        layout.bytes_per_tensor,
                    "value tensor byte offset");
  lkjai::DecoderKvCacheConfig bad{1, 1, 1, 8, 7};
  ok = ok && expect(!lkjai::decoder_kv_cache_layout(bad, &layout, &error),
                    "invalid head_dim rejected");
  if (!ok) return 1;
  std::cout << "{\"status\":\"pass\",\"kv_cache_backend\":"
            << "\"cuda_contiguous_bf16\"}\n";
  return 0;
}
