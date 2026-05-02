#include "transformer_state.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <sstream>

#include "json_min.hpp"
#include "artifact.hpp"

namespace lkjai {
namespace {

uint16_t bf16(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

void text(const std::filesystem::path& path, const std::string& body) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << body;
}

void pad(std::ofstream& out) {
  auto pos = static_cast<uint64_t>(out.tellp());
  for (uint64_t i = pos % 256; i != 0 && i < 256; ++i) out.put('\0');
}

std::string shape_json(const std::vector<int>& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out << ",";
    out << shape[i];
  }
  out << "]";
  return out.str();
}

void append_index(std::ostringstream* index, const Parameter& p, uint64_t off,
                  uint64_t bytes, bool* first,
                  const std::string& name_prefix = "",
                  const std::string& dtype = "bf16") {
  if (!*first) *index << ",";
  *first = false;
  *index << "{\"name\":\"" << json_escape(name_prefix + p.name)
         << "\",\"dtype\":\"" << dtype << "\","
         << "\"shape\":" << shape_json(p.shape) << ",\"byte_offset\":" << off
         << ",\"byte_length\":" << bytes << "}";
}

void write_param(std::ofstream& weights, std::ostringstream* index,
                 const Parameter& p, bool* first, uint64_t* hash) {
  pad(weights);
  auto off = static_cast<uint64_t>(weights.tellp());
  for (float value : p.w) {
    auto packed = bf16(value);
    weights.write(reinterpret_cast<const char*>(&packed), sizeof(packed));
    *hash = (*hash ^ packed) * 1099511628211ull;
  }
  append_index(index, p, off, static_cast<uint64_t>(weights.tellp()) - off, first);
}

void write_f32_tensor(std::ofstream& file, std::ostringstream* index,
                      const Parameter& p, const std::vector<float>& values,
                      const std::string& prefix, bool* first) {
  pad(file);
  auto off = static_cast<uint64_t>(file.tellp());
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  append_index(index, p, off, static_cast<uint64_t>(file.tellp()) - off, first,
               prefix, "f32");
}

template <typename Fn>
void params(const TransformerState& s, Fn fn) {
  fn(s.tok_embeddings);
  fn(s.pos_embeddings);
  for (const auto& l : s.layers) {
    fn(l.attn_norm);
    fn(l.q_proj);
    fn(l.k_proj);
    fn(l.v_proj);
    fn(l.o_proj);
    fn(l.mlp_norm);
    fn(l.gate_proj);
    fn(l.up_proj);
    fn(l.down_proj);
  }
  fn(s.final_norm);
  fn(s.lm_head);
}

std::string config_json(const TransformerConfig& c) {
  std::ostringstream out;
  out << "{\"model\":\"" << json_escape(c.model) << "\",\"dtype\":\"bf16\","
      << "\"vocab_size\":" << c.vocab_size << ",\"context\":" << c.context
      << ",\"layers\":" << c.layers << ",\"hidden_size\":" << c.hidden_size
      << ",\"heads\":" << c.heads << ",\"kv_heads\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim << ",\"ffn_size\":" << c.ffn_size
      << ",\"activation\":\"swiglu\",\"rope_theta\":" << c.rope_theta
      << ",\"rms_norm_eps\":" << c.rms_norm_eps << ",\"tie_embeddings\":"
      << (c.tie_embeddings ? "true" : "false") << ",\"seed\":" << c.seed
      << "}\n";
  return out.str();
}

std::string manifest_json(const std::string& artifact_kind,
                          std::string_view config,
                          std::string_view tokenizer,
                          const std::string& weights_checksum) {
  std::ostringstream out;
  out << "{\"format\":\"lkjai-native-artifact-v2\",\"kind\":\"transformer\","
      << "\"artifact_kind\":\"" << artifact_kind << "\","
      << "\"weights_checksum\":\"" << json_escape(weights_checksum) << "\","
      << "\"config_checksum\":\"" << artifact_text_checksum(config) << "\","
      << "\"tokenizer_checksum\":\"" << artifact_text_checksum(tokenizer)
      << "\"}\n";
  return out.str();
}

}  // namespace

bool write_transformer_artifact(const std::filesystem::path& dir,
                                const TransformerState& state, int step,
                                int microsteps, int batch_size, int seq_len,
                                int grad_accum, double loss, bool checkpoint,
                                std::string* checksum) {
  std::filesystem::create_directories(dir);
  std::ofstream weights(dir / "weights.lkjw", std::ios::binary);
  if (!weights) return false;
  std::ostringstream index;
  bool first = true;
  uint64_t hash = 1469598103934665603ull;
  index << "{\"tensors\":[";
  params(state, [&](const Parameter& p) {
    write_param(weights, &index, p, &first, &hash);
  });
  index << "]}\n";
  std::ostringstream weight_hash;
  weight_hash << std::hex << hash;
  *checksum = weight_hash.str();
  text(dir / "weights.index.json", index.str());
  auto config = config_json(state.cfg);
  auto tokenizer = "{\"format\":\"uint16-packed-cache\",\"vocab_size\":" +
                   std::to_string(state.cfg.vocab_size) + "}\n";
  text(dir / "manifest.json", manifest_json(checkpoint ? "checkpoint" : "export",
                                            config, tokenizer, *checksum));
  text(dir / "config.json", config);
  text(dir / "tokenizer.json", tokenizer);
  text(dir / "trainer_state.json",
       "{\"optimizer_steps\":" + std::to_string(step) +
           ",\"microsteps\":" + std::to_string(microsteps) +
           ",\"batch_size\":" + std::to_string(batch_size) +
           ",\"seq_len\":" + std::to_string(seq_len) +
           ",\"grad_accum\":" + std::to_string(grad_accum) +
           ",\"loss\":" + std::to_string(loss) +
           ",\"logits_checksum\":\"" + *checksum +
           "\",\"checkpoint\":" + (checkpoint ? "true" : "false") + "}\n");
  if (!checkpoint) return true;
  std::ofstream opt(dir / "optimizer.lkjw", std::ios::binary);
  if (!opt) return false;
  std::ostringstream opt_index;
  bool opt_first = true;
  opt_index << "{\"tensors\":[";
  params(state, [&](const Parameter& p) {
    write_f32_tensor(opt, &opt_index, p, p.w, "master.", &opt_first);
    write_f32_tensor(opt, &opt_index, p, p.m, "adam_m.", &opt_first);
    write_f32_tensor(opt, &opt_index, p, p.v, "adam_v.", &opt_first);
  });
  opt_index << "]}\n";
  text(dir / "optimizer.index.json", opt_index.str());
  return true;
}

}  // namespace lkjai
