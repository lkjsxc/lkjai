#include "dense_cuda.hpp"

#include <cmath>
#include <cstring>
#include <fstream>

#include "dense_cuda_internal.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::vector<int> parse_tokens(const std::string& csv) {
  std::vector<int> tokens;
  size_t pos = 0;
  while (pos < csv.size()) {
    size_t comma = csv.find(',', pos);
    auto part = csv.substr(pos, comma == std::string::npos
                                    ? std::string::npos
                                    : comma - pos);
    tokens.push_back(std::stoi(part));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return tokens;
}

}  // namespace

DenseConfig dense_config_from_artifact(const std::filesystem::path& dir) {
  DenseConfig cfg;
  auto config = read_text(dir / "config.json");
  cfg.model = json_first_string(config, "model");
  cfg.vocab_size = json_int_value(config, "vocab_size", cfg.vocab_size);
  cfg.context = json_int_value(config, "context", cfg.context);
  cfg.layers = json_int_value(config, "layers", cfg.layers);
  cfg.hidden_size = json_int_value(config, "hidden_size", cfg.hidden_size);
  cfg.heads = json_int_value(config, "heads", cfg.heads);
  cfg.kv_heads = json_int_value(config, "kv_heads", cfg.kv_heads);
  cfg.head_dim = json_int_value(config, "head_dim", cfg.head_dim);
  cfg.ffn_size = json_int_value(config, "ffn_size", cfg.ffn_size);
  cfg.seed = json_int_value(config, "seed", cfg.seed);
  return cfg;
}

std::vector<float> read_dense_tensor(const std::filesystem::path& dir,
                                     const std::string& name,
                                     std::string* error) {
  auto index = read_text(dir / "weights.index.json");
  auto pos = index.find("\"name\":\"" + name + "\"");
  if (pos == std::string::npos) {
    *error = "dense artifact missing tensor " + name;
    return {};
  }
  auto off_pos = index.find("\"byte_offset\":", pos);
  auto len_pos = index.find("\"byte_length\":", pos);
  if (off_pos == std::string::npos || len_pos == std::string::npos) {
    *error = "dense tensor has invalid byte range";
    return {};
  }
  uint64_t offset = std::stoull(index.substr(off_pos + 14));
  uint64_t bytes = std::stoull(index.substr(len_pos + 14));
  std::ifstream in(dir / "weights.lkjw", std::ios::binary);
  in.seekg(static_cast<std::streamoff>(offset));
  std::vector<float> out(static_cast<size_t>(bytes / sizeof(uint16_t)));
  for (float& value : out) {
    uint16_t packed = 0;
    in.read(reinterpret_cast<char*>(&packed), sizeof(packed));
    value = bf16_to_float(packed);
  }
  if (!in) *error = "failed to read dense tensor " + name;
  return out;
}

bool dense_cuda_logits_check(const std::filesystem::path& model_dir,
                             const std::string& token_csv, std::string* json,
                             std::string* error) {
  error->clear();
  DenseConfig cfg = dense_config_from_artifact(model_dir);
  auto emb = read_dense_tensor(model_dir, "tok_embeddings", error);
  if (!error->empty()) return false;
  auto head = read_dense_tensor(model_dir, "lm_head", error);
  if (!error->empty()) return false;
  auto expected = static_cast<size_t>(cfg.vocab_size * cfg.hidden_size);
  if (emb.size() != expected || head.size() != expected) {
    *error = "dense tensor shape does not match config";
    return false;
  }
  auto tokens = parse_tokens(token_csv);
  if (tokens.empty() || static_cast<int>(tokens.size()) > cfg.context) {
    *error = "token list must fit dense model context";
    return false;
  }
  int token = tokens.back() % cfg.vocab_size;
  std::vector<float> logits(static_cast<size_t>(cfg.vocab_size), 0.0f);
  auto* h = emb.data() + static_cast<size_t>(token) * cfg.hidden_size;
  for (int v = 0; v < cfg.vocab_size; ++v) {
    auto* w = head.data() + static_cast<size_t>(v) * cfg.hidden_size;
    for (int i = 0; i < cfg.hidden_size; ++i) logits[v] += h[i] * w[i];
  }
  for (float value : logits) {
    if (!std::isfinite(value)) {
      *error = "dense logits contain non-finite value";
      return false;
    }
  }
  *json = "{\"status\":\"pass\",\"kind\":\"dense\",\"shape\":[1," +
          std::to_string(cfg.vocab_size) + "],\"finite\":true,"
          "\"checksum\":\"" + dense_checksum_floats(logits) + "\"}";
  return true;
}

}  // namespace lkjai
