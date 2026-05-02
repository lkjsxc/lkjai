#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace lkjai {

struct PackedCacheStatus {
  bool ok = false;
  std::filesystem::path dir;
  uint64_t windows = 0;
  uint64_t tokens = 0;
  int sequence_len = 0;
  int vocab_size = 0;
  std::string error;
};

PackedCacheStatus inspect_packed_cache(const std::filesystem::path& dir);
bool validate_packed_cache_layout(const std::filesystem::path& dir,
                                  std::string_view metadata,
                                  uint64_t token_bytes, uint64_t mask_bytes,
                                  uint64_t start_bytes,
                                  PackedCacheStatus* status);

struct PackedBatch {
  std::vector<uint16_t> tokens;
  std::vector<uint8_t> loss_mask;
  int batch_size = 0;
  int sequence_len = 0;
};

bool load_packed_batch(const std::filesystem::path& dir, int first_window,
                       int batch_size, int sequence_len, PackedBatch* batch,
                       std::string* error);
bool migrate_packed_cache_v1_to_v2(const std::filesystem::path& in,
                                   const std::filesystem::path& out,
                                   const std::filesystem::path& config,
                                   const std::string& link_mode,
                                   std::string* error);

}  // namespace lkjai
