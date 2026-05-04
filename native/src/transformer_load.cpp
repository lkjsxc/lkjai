#include "transformer_state.hpp"

#include <bit>
#include <cstdint>
#include <fstream>

#include "json_min.hpp"

namespace lkjai {
namespace {

float f32(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

bool read_param(const std::filesystem::path& dir, const std::string& index,
                Parameter* p) {
  auto name = "\"name\":\"" + p->name + "\"";
  auto pos = index.find(name);
  if (pos == std::string::npos) return false;
  auto off_pos = index.find("\"byte_offset\":", pos);
  auto len_pos = index.find("\"byte_length\":", pos);
  if (off_pos == std::string::npos || len_pos == std::string::npos) return false;
  uint64_t off = std::stoull(index.substr(off_pos + 14));
  uint64_t len = std::stoull(index.substr(len_pos + 14));
  if (len != p->w.size() * sizeof(uint16_t)) return false;
  std::ifstream in(dir / "weights.lkjw", std::ios::binary);
  in.seekg(static_cast<std::streamoff>(off));
  for (float& value : p->w) {
    uint16_t packed = 0;
    in.read(reinterpret_cast<char*>(&packed), sizeof(packed));
    value = f32(packed);
  }
  return static_cast<bool>(in);
}

bool read_f32_tensor(const std::filesystem::path& dir, const std::string& index,
                     const std::string& name, std::vector<float>* values) {
  auto pos = index.find("\"name\":\"" + name + "\"");
  if (pos == std::string::npos) return false;
  auto off_pos = index.find("\"byte_offset\":", pos);
  auto len_pos = index.find("\"byte_length\":", pos);
  if (off_pos == std::string::npos || len_pos == std::string::npos) return false;
  uint64_t off = std::stoull(index.substr(off_pos + 14));
  uint64_t len = std::stoull(index.substr(len_pos + 14));
  if (len != values->size() * sizeof(float)) return false;
  std::ifstream in(dir / "optimizer.lkjw", std::ios::binary);
  in.seekg(static_cast<std::streamoff>(off));
  in.read(reinterpret_cast<char*>(values->data()),
          static_cast<std::streamsize>(values->size() * sizeof(float)));
  return static_cast<bool>(in);
}

template <typename Fn>
void params(TransformerState* s, Fn fn) {
  fn(&s->tok_embeddings);
  fn(&s->pos_embeddings);
  for (auto& l : s->layers) {
    fn(&l.attn_norm);
    fn(&l.q_proj);
    fn(&l.k_proj);
    fn(&l.v_proj);
    fn(&l.o_proj);
    fn(&l.mlp_norm);
    fn(&l.gate_proj);
    fn(&l.up_proj);
    fn(&l.down_proj);
  }
  fn(&s->final_norm);
  fn(&s->lm_head);
}

bool same_config(const TransformerConfig& a, const TransformerConfig& b,
                 std::string* error) {
  if (a.vocab_size != b.vocab_size || a.context != b.context ||
      a.layers != b.layers || a.hidden_size != b.hidden_size ||
      a.heads != b.heads || a.kv_heads != b.kv_heads ||
      a.head_dim != b.head_dim || a.ffn_size != b.ffn_size ||
      a.activation != b.activation || a.tie_embeddings != b.tie_embeddings ||
      a.seed != b.seed || a.kind != b.kind) {
    *error = "transformer checkpoint config mismatch";
    return false;
  }
  return true;
}

}  // namespace

bool load_transformer_artifact(const std::filesystem::path& dir,
                               TransformerState* state, std::string* error) {
  TransformerConfig cfg;
  if (!load_transformer_config(dir / "config.json", &cfg, error)) return false;
  init_transformer_state(cfg, state);
  auto index = read_text(dir / "weights.index.json");
  bool ok = true;
  params(state, [&](Parameter* p) { ok = ok && read_param(dir, index, p); });
  if (!ok) *error = "failed to load transformer artifact tensors";
  return ok;
}

bool load_transformer_checkpoint(const std::filesystem::path& dir,
                                 const TransformerConfig& requested,
                                 int batch_size, int seq_len, int grad_accum,
                                 TransformerState* state, int* steps,
                                 int* microsteps, std::string* error) {
  auto manifest = read_text(dir / "manifest.json");
  if (!(contains_json_string(manifest, "kind", "transformer") ||
        contains_json_string(manifest, "kind", "decoder")) ||
      !contains_json_string(manifest, "artifact_kind", "checkpoint")) {
    *error = "resume path must be a transformer or decoder checkpoint artifact";
    return false;
  }
  TransformerConfig checkpoint_cfg;
  if (!load_transformer_config(dir / "config.json", &checkpoint_cfg, error)) {
    return false;
  }
  if (!same_config(requested, checkpoint_cfg, error)) return false;
  auto trainer = read_text(dir / "trainer_state.json");
  *steps = json_int_value(trainer, "optimizer_steps", -1);
  *microsteps = json_int_value(trainer, "microsteps", -1);
  int ckpt_batch = json_int_value(trainer, "batch_size", -1);
  int ckpt_seq = json_int_value(trainer, "seq_len", -1);
  int ckpt_grad = json_int_value(trainer, "grad_accum", -1);
  if (*steps < 0 || *microsteps < 0 || ckpt_batch != batch_size ||
      ckpt_seq != seq_len || ckpt_grad != grad_accum) {
    *error = "transformer checkpoint trainer_state mismatch";
    return false;
  }
  init_transformer_state(checkpoint_cfg, state);
  auto index = read_text(dir / "optimizer.index.json");
  bool ok = true;
  params(state, [&](Parameter* p) {
    ok = ok && read_f32_tensor(dir, index, "master." + p->name, &p->w);
    ok = ok && read_f32_tensor(dir, index, "adam_m." + p->name, &p->m);
    ok = ok && read_f32_tensor(dir, index, "adam_v." + p->name, &p->v);
  });
  if (!ok) *error = "failed to load transformer checkpoint optimizer tensors";
  return ok;
}

}  // namespace lkjai
