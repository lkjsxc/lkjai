#include "packed_cache.hpp"

#include <fstream>

#include "json_min.hpp"
#include "packed_cache_digest.hpp"

namespace lkjai {
namespace {

bool require_digest(std::string_view metadata, std::string_view key,
                    std::string_view expected, PackedCacheStatus* status) {
  auto actual = json_first_string(metadata, key);
  if (actual.empty()) {
    status->error = "metadata missing " + std::string(key);
    return false;
  }
  if (actual != expected) {
    status->error = "metadata " + std::string(key) + " mismatch";
    return false;
  }
  return true;
}

}  // namespace

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
  status->smoke_fixture = json_bool_value(metadata, "smoke_fixture", false);
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
  int max_token_id = 0;
  std::ifstream token_file(dir / "tokens.bin", std::ios::binary);
  for (uint64_t i = 0; i < status->tokens; ++i) {
    uint16_t token = 0;
    token_file.read(reinterpret_cast<char*>(&token), sizeof(token));
    if (!token_file) {
      status->error = "failed to read tokens.bin";
      return false;
    }
    if (token >= status->vocab_size) {
      status->error = "tokens.bin contains token id outside metadata vocab_size";
      return false;
    }
    if (token > max_token_id) max_token_id = token;
  }
  auto expected_max = json_int_value(metadata, "max_token_id", -1);
  if (expected_max >= 0 && expected_max != max_token_id) {
    status->error = "metadata max_token_id mismatch";
    return false;
  }
  if (!status->smoke_fixture) {
    if (!require_digest(metadata, "tokens_checksum",
                        packed_file_digest(dir / "tokens.bin"), status)) return false;
    if (!require_digest(metadata, "loss_mask_checksum",
                        packed_file_digest(dir / "loss_mask.bin"), status)) return false;
    if (!require_digest(metadata, "starts_checksum",
                        packed_file_digest(dir / "starts.bin"), status)) return false;
    if (!require_digest(metadata, "packed_data_checksum",
                        packed_payload_digest(dir), status)) return false;
    for (const auto& key : {"tokenizer_digest", "config_digest", "source_digest"}) {
      if (json_first_string(metadata, key).empty()) {
        status->error = "metadata missing " + std::string(key);
        return false;
      }
    }
  }
  return true;
}

}  // namespace lkjai
