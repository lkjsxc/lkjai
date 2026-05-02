#include "packed_cache.hpp"

#include <filesystem>

#include "json_min.hpp"

namespace lkjai {
namespace {

bool require_file(const std::filesystem::path& path, std::string* error) {
  if (std::filesystem::is_regular_file(path)) return true;
  *error = "missing " + path.filename().string();
  return false;
}

uint64_t byte_size(const std::filesystem::path& path) {
  return static_cast<uint64_t>(std::filesystem::file_size(path));
}

}  // namespace

PackedCacheStatus inspect_packed_cache(const std::filesystem::path& dir) {
  PackedCacheStatus status;
  status.dir = dir;
  const auto metadata = dir / "metadata.json";
  const auto tokens = dir / "tokens.bin";
  const auto mask = dir / "loss_mask.bin";
  const auto starts = dir / "starts.bin";
  for (const auto& path : {metadata, tokens, mask, starts}) {
    if (!require_file(path, &status.error)) return status;
  }
  auto meta = read_text(metadata);
  if (!contains_json_string(meta, "format", "lkjai-packed-cache-v2")) {
    status.error = "metadata format must be lkjai-packed-cache-v2";
    return status;
  }
  auto token_bytes = byte_size(tokens);
  auto mask_bytes = byte_size(mask);
  auto start_bytes = byte_size(starts);
  if (token_bytes == 0 || token_bytes % 2 != 0) {
    status.error = "tokens.bin must contain uint16 tokens";
    return status;
  }
  if (mask_bytes != token_bytes / 2) {
    status.error = "loss_mask.bin must match token count";
    return status;
  }
  if (start_bytes == 0 || start_bytes % 8 != 0) {
    status.error = "starts.bin must contain uint64 offsets";
    return status;
  }
  status.ok = true;
  status.tokens = token_bytes / 2;
  status.windows = start_bytes / 8;
  return status;
}

}  // namespace lkjai
