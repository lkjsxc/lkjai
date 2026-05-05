#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
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
  bool smoke_fixture = false;
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

class PackedCacheReader {
 public:
  PackedCacheReader() = default;
  PackedCacheReader(const PackedCacheReader&) = delete;
  PackedCacheReader& operator=(const PackedCacheReader&) = delete;

  bool open(const std::filesystem::path& dir, int sequence_len,
            int max_vocab_size, std::string* error);
  bool load_batch(uint64_t first_window, int batch_size, PackedBatch* batch,
                  std::string* error);
  bool load_batch_into(uint64_t first_window, int batch_size, uint16_t* tokens,
                       uint8_t* mask, std::string* error);
  const PackedCacheStatus& status() const { return status_; }

 private:
  PackedCacheStatus status_;
  std::ifstream starts_;
  std::ifstream tokens_;
  std::ifstream mask_;
};

bool migrate_packed_cache_v1_to_v2(const std::filesystem::path& in,
                                   const std::filesystem::path& out,
                                   const std::filesystem::path& config,
                                   const std::string& link_mode,
                                   std::string* error);
bool packed_cache_allowed_for_run(const PackedCacheStatus& status,
                                  const std::string& run_purpose,
                                  std::string* error);

}  // namespace lkjai
