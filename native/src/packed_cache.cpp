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
  if (!contains_json_string(meta, "format", "lkjai-packed-cache-v2")) {
    status.error = "metadata format must be lkjai-packed-cache-v2";
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

bool migrate_packed_cache_v1_to_v2(const std::filesystem::path& in,
                                   const std::filesystem::path& out,
                                   const std::filesystem::path& config,
                                   const std::string& link_mode,
                                   std::string* error) {
  auto meta = read_text(in / "metadata.json");
  if (!contains_json_string(meta, "format", "lkjai-packed-cache-v1")) {
    *error = "input metadata format must be lkjai-packed-cache-v1";
    return false;
  }
  auto cfg = read_text(config);
  int sequence_len = json_int_value(meta, "sequence_len", 0);
  int vocab_size = json_int_value(meta, "vocab_size", 0);
  int cfg_context = json_int_value(cfg, "context", 0);
  int cfg_vocab = json_int_value(cfg, "vocab_size", 0);
  if (sequence_len <= 1 || vocab_size <= 0) {
    *error = "v1 metadata has invalid sequence_len or vocab_size";
    return false;
  }
  if (cfg_context > 0 && sequence_len > cfg_context) {
    *error = "v1 sequence_len exceeds transformer context";
    return false;
  }
  if (cfg_vocab > 0 && vocab_size > cfg_vocab) {
    *error = "v1 vocab_size exceeds transformer vocab_size";
    return false;
  }
  PackedCacheStatus raw;
  raw.dir = in;
  auto tokens = in / "tokens.bin";
  auto mask = in / "loss_mask.bin";
  auto starts = in / "starts.bin";
  for (const auto& path : {tokens, mask, starts}) {
    if (!require_file(path, error)) return false;
  }
  auto token_bytes = byte_size(tokens);
  auto mask_bytes = byte_size(mask);
  auto start_bytes = byte_size(starts);
  if (token_bytes == 0 || token_bytes % 2 != 0 ||
      mask_bytes != token_bytes / 2 || start_bytes == 0 ||
      start_bytes % 8 != 0) {
    *error = "v1 binary layout is not compatible with packed-cache v2";
    return false;
  }
  std::filesystem::create_directories(out);
  for (const auto& name : {"tokens.bin", "loss_mask.bin", "starts.bin"}) {
    auto src = in / name;
    auto dst = out / name;
    std::filesystem::remove(dst);
    if (link_mode == "hardlink") {
      std::filesystem::create_hard_link(src, dst);
    } else {
      std::filesystem::copy_file(src, dst);
    }
  }
  std::ofstream(out / "metadata.json")
      << "{\"format\":\"lkjai-packed-cache-v2\",\"migrated_from\":\"v1\","
      << "\"smoke_fixture\":true,"
      << "\"sequence_len\":" << sequence_len << ",\"vocab_size\":"
      << vocab_size << ",\"token_dtype\":\"uint16\",\"token_count\":"
      << (token_bytes / 2) << ",\"row_count\":" << (start_bytes / 8)
      << "}\n";
  return true;
}

bool packed_cache_allowed_for_run(const PackedCacheStatus& status,
                                  const std::string& run_purpose,
                                  std::string* error) {
  if (!status.smoke_fixture || run_purpose == "smoke") return true;
  *error = "smoke packed cache fixture cannot be used for real training";
  return false;
}

}  // namespace lkjai
