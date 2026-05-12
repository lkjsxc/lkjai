#include <iostream>
#include <string>
#include <vector>

#include "decoder_kv_cache.hpp"
#include "runtime_device.hpp"

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
  lkjai::DecoderKvCache cache;
  ok = ok && expect(lkjai::decoder_kv_cache_allocate(cfg, &cache, &error),
                    error);
  ok = ok && expect(cache.allocated_bytes == layout.total_bytes,
                    "allocation accounting");
  ok = ok && expect(lkjai::decoder_kv_cache_write(
                        &cache, false, 0, 0, 0, 0, 1, 42, &error),
                    error);
  uint16_t value = 0;
  ok = ok && expect(lkjai::decoder_kv_cache_read(
                        cache, false, 0, 0, 0, 0, 1, &value, &error),
                    error);
  ok = ok && expect(value == 42, "write/read value");
  std::vector<uint16_t> keys(static_cast<size_t>(cfg.layers * cfg.kv_heads *
                                                cfg.head_dim),
                             7);
  std::vector<uint16_t> vals(keys.size(), 9);
  ok = ok && expect(lkjai::decoder_kv_cache_append(&cache, 1, keys, vals,
                                                   &error),
                    error);
  ok = ok && expect(cache.next_position[1] == 1, "append advanced position");
  ok = ok && expect(lkjai::decoder_kv_cache_read(
                        cache, true, 0, 1, 0, 0, 0, &value, &error),
                    error);
  ok = ok && expect(value == 9, "append/read value");
  std::vector<float> key_f(static_cast<size_t>(2 * cfg.kv_heads * cfg.head_dim),
                           1.0f);
  std::vector<float> val_f(key_f.size(), 2.0f);
  lkjai::CudaExecutionContext ctx;
  lkjai::DeviceTensor dk(
      {lkjai::DeviceDType::bf16, {2, cfg.kv_heads, cfg.head_dim}},
      ctx.stream());
  lkjai::DeviceTensor dv(
      {lkjai::DeviceDType::bf16, {2, cfg.kv_heads, cfg.head_dim}},
      ctx.stream());
  dk.copy_from_host_f32(key_f, ctx.stream());
  dv.copy_from_host_f32(val_f, ctx.stream());
  ok = ok && expect(lkjai::decoder_kv_cache_append_device_layer(
                        &cache, cfg.layers - 1, 1, dk.data(), dv.data(), 1,
                        2, ctx.stream(), &error),
                    error);
  ok = ok && expect(cache.next_position[0] == 3,
                    "device append advanced position");
  ok = ok && expect(lkjai::decoder_kv_cache_read(
                        cache, false, cfg.layers - 1, 0, 0, 2, 0, &value,
                        &error),
                    error);
  ok = ok && expect(value == 0x3f80, "device key append value");
  ok = ok && expect(lkjai::decoder_kv_cache_read(
                        cache, true, cfg.layers - 1, 0, 1, 2,
                        cfg.head_dim - 1, &value, &error),
                    error);
  ok = ok && expect(value == 0x4000, "device value append value");
  lkjai::DecoderKvCacheConfig bad{1, 1, 1, 8, 7};
  ok = ok && expect(!lkjai::decoder_kv_cache_layout(bad, &layout, &error),
                    "invalid head_dim rejected");
  if (!ok) return 1;
  std::cout << "{\"status\":\"pass\",\"kv_cache_backend\":"
            << "\"cuda_contiguous_bf16\"}\n";
  return 0;
}
