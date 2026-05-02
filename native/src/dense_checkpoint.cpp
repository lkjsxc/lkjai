#include "dense_train_internal.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>

#include "artifact.hpp"
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

bool same_dense_config(const DenseConfig& a, const DenseConfig& b,
                       std::string* error) {
  if (a.model != b.model) {
    *error = "resume config model mismatch";
    return false;
  }
  if (a.vocab_size != b.vocab_size || a.context != b.context ||
      a.layers != b.layers || a.hidden_size != b.hidden_size ||
      a.heads != b.heads || a.kv_heads != b.kv_heads ||
      a.head_dim != b.head_dim || a.ffn_size != b.ffn_size) {
    *error = "resume config shape mismatch";
    return false;
  }
  if (a.seed != b.seed) {
    *error = "resume config seed mismatch";
    return false;
  }
  return true;
}

bool read_indexed_f32_tensor(const std::filesystem::path& dir,
                             const std::string& name, size_t elements,
                             std::vector<float>* out, std::string* error) {
  auto entry = index_entry(read_text(dir / "optimizer.index.json"), name);
  if (entry.empty()) {
    *error = "checkpoint optimizer missing tensor " + name;
    return false;
  }
  if (!contains_json_string(entry, "dtype", "f32")) {
    *error = "checkpoint optimizer tensor is not f32: " + name;
    return false;
  }
  uint64_t offset = 0;
  uint64_t bytes = 0;
  if (!read_u64_field(entry, "\"byte_offset\"", &offset) ||
      !read_u64_field(entry, "\"byte_length\"", &bytes)) {
    *error = "checkpoint optimizer tensor has invalid byte range: " + name;
    return false;
  }
  if (bytes != static_cast<uint64_t>(elements * sizeof(float))) {
    *error = "checkpoint optimizer tensor shape mismatch: " + name;
    return false;
  }
  auto path = dir / "optimizer.lkjw";
  if (!std::filesystem::is_regular_file(path) ||
      offset > std::filesystem::file_size(path) ||
      bytes > std::filesystem::file_size(path) - offset) {
    *error = "checkpoint optimizer tensor range outside optimizer.lkjw: " + name;
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(offset));
  out->assign(elements, 0.0f);
  in.read(reinterpret_cast<char*>(out->data()),
          static_cast<std::streamsize>(bytes));
  if (!in) *error = "failed to read checkpoint optimizer tensor " + name;
  return static_cast<bool>(in);
}

double json_double_field(std::string_view text, std::string_view key) {
  auto pos = text.find("\"" + std::string(key) + "\"");
  if (pos == std::string_view::npos) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  pos = text.find(':', pos);
  if (pos == std::string_view::npos) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  try {
    return std::stod(std::string(text.substr(pos + 1)));
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace

bool load_dense_checkpoint(const std::filesystem::path& dir,
                           const DenseConfig& requested, int batch_size,
                           int seq_len, int grad_accum,
                           DenseTrainState* state,
                           DenseCheckpointMetadata* metadata,
                           std::string* error) {
  if (!inspect_artifact(dir, error)) return false;
  auto manifest = read_text(dir / "manifest.json");
  if (!contains_json_string(manifest, "kind", "dense") ||
      !contains_json_string(manifest, "artifact_kind", "checkpoint")) {
    *error = "resume path must be a dense checkpoint artifact";
    return false;
  }
  DenseConfig checkpoint_cfg;
  if (!load_dense_config(dir / "config.json", &checkpoint_cfg, error)) {
    return false;
  }
  if (!same_dense_config(requested, checkpoint_cfg, error)) return false;
  auto trainer = read_text(dir / "trainer_state.json");
  metadata->optimizer_steps = json_int_value(trainer, "optimizer_steps", -1);
  metadata->microsteps = json_int_value(trainer, "microsteps", -1);
  metadata->batch_size = json_int_value(trainer, "batch_size", -1);
  metadata->seq_len = json_int_value(trainer, "seq_len", -1);
  metadata->grad_accum = json_int_value(trainer, "grad_accum", -1);
  metadata->loss = json_double_field(trainer, "loss");
  metadata->logits_checksum = json_first_string(trainer, "logits_checksum");
  if (metadata->optimizer_steps < 0 || metadata->microsteps < 0 ||
      metadata->batch_size <= 0 || metadata->seq_len <= 0 ||
      metadata->grad_accum <= 0 || !std::isfinite(metadata->loss) ||
      metadata->logits_checksum.empty()) {
    *error = "resume trainer_state.json is missing required fields";
    return false;
  }
  if (metadata->batch_size != batch_size || metadata->seq_len != seq_len ||
      metadata->grad_accum != grad_accum) {
    *error = "resume trainer_state.json batch/seq/grad settings mismatch";
    return false;
  }
  if (metadata->microsteps != metadata->optimizer_steps * metadata->grad_accum) {
    *error = "resume trainer_state.json microsteps do not match optimizer state";
    return false;
  }
  state->cfg = requested;
  const auto elements =
      static_cast<size_t>(requested.vocab_size * requested.hidden_size);
  state->grad_emb.assign(elements, 0.0f);
  state->grad_head.assign(elements, 0.0f);
  return read_indexed_f32_tensor(dir, "master.tok_embeddings", elements,
                                 &state->emb, error) &&
         read_indexed_f32_tensor(dir, "adam_m.tok_embeddings", elements,
                                 &state->m_emb, error) &&
         read_indexed_f32_tensor(dir, "adam_v.tok_embeddings", elements,
                                 &state->v_emb, error) &&
         read_indexed_f32_tensor(dir, "master.lm_head", elements,
                                 &state->head, error) &&
         read_indexed_f32_tensor(dir, "adam_m.lm_head", elements,
                                 &state->m_head, error) &&
         read_indexed_f32_tensor(dir, "adam_v.lm_head", elements,
                                 &state->v_head, error);
}

}  // namespace lkjai
