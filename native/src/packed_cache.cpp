#include "packed_cache.hpp"

#include <filesystem>
#include <fstream>

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
  status.sequence_len = json_int_value(meta, "sequence_len", 0);
  status.vocab_size = json_int_value(meta, "vocab_size", 0);
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

bool load_packed_batch(const std::filesystem::path& dir, int first_window,
                       int batch_size, int sequence_len, PackedBatch* batch,
                       std::string* error) {
  auto status = inspect_packed_cache(dir);
  if (!status.ok) {
    *error = status.error;
    return false;
  }
  if (sequence_len <= 1) {
    *error = "sequence length must be greater than 1";
    return false;
  }
  if (batch_size <= 0 || first_window < 0) {
    *error = "invalid packed batch range";
    return false;
  }
  std::ifstream starts(dir / "starts.bin", std::ios::binary);
  std::ifstream tokens(dir / "tokens.bin", std::ios::binary);
  std::ifstream mask(dir / "loss_mask.bin", std::ios::binary);
  if (!starts || !tokens || !mask) {
    *error = "failed to open packed cache files";
    return false;
  }
  batch->tokens.assign(static_cast<size_t>(batch_size * sequence_len), 0);
  batch->loss_mask.assign(static_cast<size_t>(batch_size * sequence_len), 0);
  batch->batch_size = batch_size;
  batch->sequence_len = sequence_len;
  for (int row = 0; row < batch_size; ++row) {
    auto window = static_cast<uint64_t>(
        (first_window + row) % static_cast<int>(status.windows));
    uint64_t offset = 0;
    starts.seekg(static_cast<std::streamoff>(window * sizeof(uint64_t)));
    starts.read(reinterpret_cast<char*>(&offset), sizeof(offset));
    if (!starts) {
      *error = "failed to read packed window offset";
      return false;
    }
    if (offset + static_cast<uint64_t>(sequence_len) > status.tokens) {
      *error = "packed window exceeds token file";
      return false;
    }
    auto token_pos = static_cast<std::streamoff>(offset * sizeof(uint16_t));
    auto mask_pos = static_cast<std::streamoff>(offset);
    auto base = static_cast<size_t>(row * sequence_len);
    tokens.seekg(token_pos);
    tokens.read(reinterpret_cast<char*>(batch->tokens.data() + base),
                static_cast<std::streamsize>(sequence_len * sizeof(uint16_t)));
    mask.seekg(mask_pos);
    mask.read(reinterpret_cast<char*>(batch->loss_mask.data() + base),
              static_cast<std::streamsize>(sequence_len));
    if (!tokens || !mask) {
      *error = "failed to read packed batch payload";
      return false;
    }
  }
  return true;
}

}  // namespace lkjai
