#include "dense_cuda.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "artifact.hpp"
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

bool read_u64_field(std::string_view text, std::string_view key, uint64_t* out) {
  auto pos = text.find(key);
  if (pos == std::string_view::npos) return false;
  pos = text.find(':', pos + key.size());
  if (pos == std::string_view::npos) return false;
  ++pos;
  while (pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  try {
    *out = static_cast<uint64_t>(std::stoull(std::string(text.substr(pos))));
    return true;
  } catch (...) {
    return false;
  }
}

std::string index_entry(std::string_view index, const std::string& name) {
  auto name_pos = index.find("\"" + name + "\"");
  if (name_pos == std::string_view::npos) return "";
  auto start = index.rfind('{', name_pos);
  auto end = index.find('}', name_pos);
  if (start == std::string_view::npos || end == std::string_view::npos ||
      end < start) {
    return "";
  }
  return std::string(index.substr(start, end - start + 1));
}

std::vector<float> read_checkpoint_tensor(const std::filesystem::path& dir,
                                          const std::string& name,
                                          size_t elements,
                                          std::string* error) {
  auto entry = index_entry(read_text(dir / "optimizer.index.json"), name);
  if (entry.empty()) {
    *error = "reference checkpoint missing tensor " + name;
    return {};
  }
  if (!contains_json_string(entry, "dtype", "f32")) {
    *error = "reference checkpoint tensor is not f32: " + name;
    return {};
  }
  uint64_t offset = 0;
  uint64_t bytes = 0;
  if (!read_u64_field(entry, "\"byte_offset\"", &offset) ||
      !read_u64_field(entry, "\"byte_length\"", &bytes) ||
      bytes != static_cast<uint64_t>(elements * sizeof(float))) {
    *error = "reference checkpoint tensor shape mismatch: " + name;
    return {};
  }
  auto path = dir / "optimizer.lkjw";
  if (!std::filesystem::is_regular_file(path) ||
      offset > std::filesystem::file_size(path) ||
      bytes > std::filesystem::file_size(path) - offset) {
    *error = "reference checkpoint tensor range outside optimizer.lkjw: " + name;
    return {};
  }
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(offset));
  std::vector<float> out(elements);
  in.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(bytes));
  if (!in) {
    *error = "failed to read reference checkpoint tensor " + name;
    return {};
  }
  return out;
}

bool logits_for_tokens(const DenseConfig& cfg, const std::vector<float>& emb,
                       const std::vector<float>& head,
                       const std::string& token_csv,
                       std::vector<float>* logits, std::string* error) {
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
  logits->assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
  auto* h = emb.data() + static_cast<size_t>(token) * cfg.hidden_size;
  for (int v = 0; v < cfg.vocab_size; ++v) {
    auto* w = head.data() + static_cast<size_t>(v) * cfg.hidden_size;
    for (int i = 0; i < cfg.hidden_size; ++i) (*logits)[v] += h[i] * w[i];
  }
  for (float value : *logits) {
    if (!std::isfinite(value)) {
      *error = "dense logits contain non-finite value";
      return false;
    }
  }
  return true;
}

std::string logits_json(const DenseConfig& cfg, const std::vector<float>& logits,
                        const std::string& reference_status,
                        double max_abs_diff, double mean_abs_diff,
                        double tolerance) {
  std::ostringstream out;
  out << "{\"status\":\"pass\",\"kind\":\"dense\",\"shape\":[1,"
      << cfg.vocab_size << "],\"finite\":true,\"checksum\":\""
      << dense_checksum_floats(logits)
      << "\",\"validation_target\":\"exported_bf16_weights\""
      << ",\"reference_check\":\"" << reference_status
      << "\",\"max_abs_diff\":" << max_abs_diff
      << ",\"mean_abs_diff\":" << mean_abs_diff
      << ",\"tolerance\":" << tolerance << "}";
  return out.str();
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
  std::vector<float> logits;
  if (!logits_for_tokens(cfg, emb, head, token_csv, &logits, error)) return false;
  *json = logits_json(cfg, logits, "not_requested", 0.0, 0.0, 0.0);
  return true;
}

bool dense_cuda_logits_check_against_checkpoint(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& reference_checkpoint,
    const std::string& token_csv, std::string* json, std::string* error) {
  error->clear();
  std::string inspect_error;
  if (!inspect_artifact(reference_checkpoint, &inspect_error)) {
    *error = "invalid reference checkpoint: " + inspect_error;
    return false;
  }
  auto manifest = read_text(reference_checkpoint / "manifest.json");
  if (!contains_json_string(manifest, "kind", "dense") ||
      !contains_json_string(manifest, "artifact_kind", "checkpoint")) {
    *error = "reference checkpoint must be a dense checkpoint artifact";
    return false;
  }
  DenseConfig export_cfg = dense_config_from_artifact(model_dir);
  DenseConfig ref_cfg = dense_config_from_artifact(reference_checkpoint);
  if (export_cfg.vocab_size != ref_cfg.vocab_size ||
      export_cfg.hidden_size != ref_cfg.hidden_size ||
      export_cfg.context != ref_cfg.context) {
    *error = "reference checkpoint config does not match export";
    return false;
  }
  auto emb = read_dense_tensor(model_dir, "tok_embeddings", error);
  if (!error->empty()) return false;
  auto head = read_dense_tensor(model_dir, "lm_head", error);
  if (!error->empty()) return false;
  std::vector<float> export_logits;
  if (!logits_for_tokens(export_cfg, emb, head, token_csv, &export_logits,
                         error)) {
    return false;
  }
  auto elements =
      static_cast<size_t>(ref_cfg.vocab_size * ref_cfg.hidden_size);
  auto ref_emb = read_checkpoint_tensor(reference_checkpoint,
                                        "master.tok_embeddings", elements,
                                        error);
  if (!error->empty()) return false;
  auto ref_head = read_checkpoint_tensor(reference_checkpoint, "master.lm_head",
                                         elements, error);
  if (!error->empty()) return false;
  std::vector<float> ref_logits;
  if (!logits_for_tokens(ref_cfg, ref_emb, ref_head, token_csv, &ref_logits,
                         error)) {
    return false;
  }
  double max_abs = 0.0;
  double sum_abs = 0.0;
  for (size_t i = 0; i < export_logits.size(); ++i) {
    double diff = std::fabs(static_cast<double>(export_logits[i]) -
                            static_cast<double>(ref_logits[i]));
    max_abs = std::max(max_abs, diff);
    sum_abs += diff;
  }
  double mean_abs = export_logits.empty()
                        ? 0.0
                        : sum_abs / static_cast<double>(export_logits.size());
  constexpr double kTolerance = 1.0e-2;
  if (max_abs > kTolerance) {
    *json = logits_json(export_cfg, export_logits, "fail", max_abs, mean_abs,
                        kTolerance);
    *error = "dense BF16 export logits exceed FP32 checkpoint tolerance";
    return false;
  }
  *json =
      logits_json(export_cfg, export_logits, "pass", max_abs, mean_abs,
                  kTolerance);
  return true;
}

}  // namespace lkjai
