#include "packed_cache.hpp"

#include <fstream>

#include "json_min.hpp"

namespace lkjai {

bool validate_packed_cache_layout(const std::filesystem::path& dir,
                                  std::string_view metadata,
                                  uint64_t token_bytes, uint64_t mask_bytes,
                                  uint64_t start_bytes,
                                  PackedCacheStatus* status) {
  if (!contains_json_string(metadata, "token_dtype", "uint16")) {
    status->error = "metadata token_dtype must be uint16";
    return false;
  }
  status->sequence_len = json_int_value(metadata, "sequence_len", 0);
  status->vocab_size = json_int_value(metadata, "vocab_size", 0);
  auto expected_tokens = json_int_value(metadata, "token_count", -1);
  auto expected_rows = json_int_value(metadata, "row_count", -1);
  if (status->sequence_len <= 1 || status->vocab_size <= 0) {
    status->error = "metadata sequence_len and vocab_size must be positive";
    return false;
  }
  if (token_bytes == 0 || token_bytes % 2 != 0) {
    status->error = "tokens.bin must contain uint16 tokens";
    return false;
  }
  if (mask_bytes != token_bytes / 2) {
    status->error = "loss_mask.bin must match token count";
    return false;
  }
  if (start_bytes == 0 || start_bytes % 8 != 0) {
    status->error = "starts.bin must contain uint64 offsets";
    return false;
  }
  status->tokens = token_bytes / 2;
  status->windows = start_bytes / 8;
  if (expected_tokens >= 0 && static_cast<uint64_t>(expected_tokens) != status->tokens) {
    status->error = "metadata token_count does not match tokens.bin";
    return false;
  }
  if (expected_rows >= 0 && static_cast<uint64_t>(expected_rows) != status->windows) {
    status->error = "metadata row_count does not match starts.bin";
    return false;
  }
  std::ifstream starts(dir / "starts.bin", std::ios::binary);
  for (uint64_t i = 0; i < status->windows; ++i) {
    uint64_t offset = 0;
    starts.read(reinterpret_cast<char*>(&offset), sizeof(offset));
    if (!starts || offset + status->sequence_len > status->tokens) {
      status->error = "starts.bin contains out-of-bounds window";
      return false;
    }
  }
  return true;
}

}  // namespace lkjai
