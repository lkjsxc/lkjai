#include "dense_cuda.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

#include "artifact.hpp"
#include "dense_cuda_internal.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

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

bool load_reference_logits(const std::filesystem::path& checkpoint,
                           const DenseConfig& cfg,
                           const std::string& token_csv,
                           std::vector<float>* logits, std::string* error) {
  auto elements = static_cast<size_t>(cfg.vocab_size * cfg.hidden_size);
  auto emb = read_checkpoint_tensor(checkpoint, "master.tok_embeddings",
                                    elements, error);
  if (!error->empty()) return false;
  auto head = read_checkpoint_tensor(checkpoint, "master.lm_head", elements,
                                     error);
  if (!error->empty()) return false;
  return dense_logits_for_tokens(cfg, emb, head, token_csv, logits, error);
}

}  // namespace

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
  if (!dense_logits_for_tokens(export_cfg, emb, head, token_csv,
                               &export_logits, error)) return false;
  std::vector<float> ref_logits;
  if (!load_reference_logits(reference_checkpoint, ref_cfg, token_csv,
                             &ref_logits, error)) return false;
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
    *json = dense_logits_check_json(export_cfg, export_logits, "fail", max_abs,
                                    mean_abs, kTolerance);
    *error = "dense BF16 export logits exceed FP32 checkpoint tolerance";
    return false;
  }
  *json = dense_logits_check_json(export_cfg, export_logits, "pass", max_abs,
                                  mean_abs, kTolerance);
  return true;
}

}  // namespace lkjai
