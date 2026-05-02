#include "packed_cache.hpp"

namespace lkjai {

bool PackedCacheReader::open(const std::filesystem::path& dir, int sequence_len,
                             int max_vocab_size, std::string* error) {
  status_ = inspect_packed_cache(dir);
  if (!status_.ok) {
    *error = status_.error;
    return false;
  }
  if (sequence_len <= 1 || sequence_len != status_.sequence_len) {
    *error = "requested seq_len must match packed cache sequence_len";
    return false;
  }
  if (max_vocab_size > 0 && status_.vocab_size > max_vocab_size) {
    *error = "packed cache vocab_size exceeds dense config vocab_size";
    return false;
  }
  starts_.open(dir / "starts.bin", std::ios::binary);
  tokens_.open(dir / "tokens.bin", std::ios::binary);
  mask_.open(dir / "loss_mask.bin", std::ios::binary);
  if (!starts_ || !tokens_ || !mask_) {
    *error = "failed to open packed cache files";
    return false;
  }
  return true;
}

bool PackedCacheReader::load_batch(uint64_t first_window, int batch_size,
                                   PackedBatch* batch, std::string* error) {
  if (!status_.ok || !starts_ || !tokens_ || !mask_) {
    *error = "packed cache reader is not open";
    return false;
  }
  if (batch_size <= 0) {
    *error = "invalid packed batch range";
    return false;
  }
  batch->tokens.assign(
      static_cast<size_t>(batch_size * status_.sequence_len), 0);
  batch->loss_mask.assign(
      static_cast<size_t>(batch_size * status_.sequence_len), 0);
  batch->batch_size = batch_size;
  batch->sequence_len = status_.sequence_len;
  for (int row = 0; row < batch_size; ++row) {
    uint64_t window =
        (first_window + static_cast<uint64_t>(row)) % status_.windows;
    uint64_t offset = 0;
    starts_.clear();
    starts_.seekg(static_cast<std::streamoff>(window * sizeof(uint64_t)));
    starts_.read(reinterpret_cast<char*>(&offset), sizeof(offset));
    if (!starts_) {
      *error = "failed to read packed window offset";
      return false;
    }
    if (offset + static_cast<uint64_t>(status_.sequence_len) > status_.tokens) {
      *error = "packed window exceeds token file";
      return false;
    }
    auto base = static_cast<size_t>(row * status_.sequence_len);
    tokens_.clear();
    mask_.clear();
    tokens_.seekg(static_cast<std::streamoff>(offset * sizeof(uint16_t)));
    mask_.seekg(static_cast<std::streamoff>(offset));
    tokens_.read(reinterpret_cast<char*>(batch->tokens.data() + base),
                 static_cast<std::streamsize>(status_.sequence_len *
                                              sizeof(uint16_t)));
    mask_.read(reinterpret_cast<char*>(batch->loss_mask.data() + base),
               static_cast<std::streamsize>(status_.sequence_len));
    if (!tokens_ || !mask_) {
      *error = "failed to read packed batch payload";
      return false;
    }
  }
  return true;
}

}  // namespace lkjai
