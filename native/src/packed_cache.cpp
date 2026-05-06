#include "packed_cache.hpp"

#include <filesystem>
#include <fstream>

#include "json_min.hpp"
#include "packed_cache_digest.hpp"

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
  if (!contains_json_string(meta, "format", "lkjai-packed-cache")) {
    status.error = "metadata format must be lkjai-packed-cache";
    return status;
  }
  auto token_bytes = byte_size(tokens);
  auto mask_bytes = byte_size(mask);
  auto start_bytes = byte_size(starts);
  if (!validate_packed_cache_layout(dir, meta, token_bytes, mask_bytes,
                                    start_bytes, &status)) return status;
  status.ok = true;
  return status;
}

bool load_packed_batch(const std::filesystem::path& dir, int first_window,
                       int batch_size, int sequence_len, PackedBatch* batch,
                       std::string* error) {
  if (first_window < 0) {
    *error = "invalid packed batch range";
    return false;
  }
  PackedCacheReader reader;
  return reader.open(dir, sequence_len, 0, error) &&
         reader.load_batch(static_cast<uint64_t>(first_window), batch_size,
                           batch, error);
}

bool packed_cache_allowed_for_run(const PackedCacheStatus& status,
                                  const std::string& run_purpose,
                                  std::string* error) {
  if (!status.smoke_fixture || run_purpose == "smoke") return true;
  *error = "smoke packed cache fixture cannot be used for real training";
  return false;
}

}  // namespace lkjai
