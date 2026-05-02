#include "transformer_train.hpp"

#include <cmath>
#include <cstdint>

#include "packed_cache.hpp"
#include "transformer_state.hpp"

namespace lkjai {

bool transformer_logits_check(const std::filesystem::path& model_dir,
                              const std::string& token_csv,
                              std::string* json, std::string* error) {
  TransformerState state;
  if (!load_transformer_artifact(model_dir, &state, error)) return false;
  PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 0;
  size_t pos = 0;
  while (pos < token_csv.size()) {
    size_t comma = token_csv.find(',', pos);
    auto part = token_csv.substr(pos, comma == std::string::npos
                                          ? std::string::npos
                                          : comma - pos);
    batch.tokens.push_back(static_cast<uint16_t>(std::stoi(part)));
    batch.loss_mask.push_back(1);
    ++batch.sequence_len;
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (batch.sequence_len < 1 || batch.sequence_len > state.cfg.context) {
    *error = "token list must fit model context";
    return false;
  }
  auto fwd = transformer_forward(batch, state);
  if (fwd.next_logits.size() != static_cast<size_t>(state.cfg.vocab_size)) {
    *error = "logits shape mismatch";
    return false;
  }
  for (float v : fwd.next_logits) {
    if (!std::isfinite(v)) {
      *error = "logits contain non-finite value";
      return false;
    }
  }
  *json = "{\"status\":\"pass\",\"shape\":[1," +
          std::to_string(state.cfg.vocab_size) + "],\"finite\":true,"
          "\"checksum\":\"" + checksum_logits(fwd.next_logits) + "\"}";
  return true;
}

}  // namespace lkjai
