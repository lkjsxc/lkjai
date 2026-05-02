#include "transformer_state.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <sstream>

#include "json_min.hpp"

namespace lkjai {
namespace {

uint16_t bf16(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

float f32(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
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
                  uint64_t bytes, bool* first) {
  if (!*first) *index << ",";
  *first = false;
  *index << "{\"name\":\"" << json_escape(p.name) << "\",\"dtype\":\"bf16\","
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
  append_index(index, p, off, static_cast<uint64_t>(weights.tellp()) - off,
               first);
}

template <typename Fn>
void params(const TransformerState& s, Fn fn) {
  fn(s.tok_embeddings);
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

}  // namespace

bool write_transformer_artifact(const std::filesystem::path& dir,
                                const TransformerState& state, int step,
                                double loss, bool checkpoint,
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
  *checksum = checksum_logits({static_cast<float>(hash & 0xffffu)});
  text(dir / "weights.index.json", index.str());
  text(dir / "manifest.json",
       "{\"format\":\"lkjai-native-artifact-v2\",\"kind\":\"transformer\"}\n");
  text(dir / "config.json", config_json(state.cfg));
  text(dir / "tokenizer.json",
       "{\"format\":\"uint16-packed-cache\",\"vocab_size\":" +
           std::to_string(state.cfg.vocab_size) + "}\n");
  text(dir / "trainer_state.json",
       "{\"optimizer_steps\":" + std::to_string(step) +
           ",\"loss\":" + std::to_string(loss) +
           ",\"logits_checksum\":\"" + *checksum +
           "\",\"checkpoint\":" + (checkpoint ? "true" : "false") + "}\n");
  if (!checkpoint) return true;
  std::ofstream opt(dir / "optimizer.lkjw", std::ios::binary);
  if (!opt) return false;
  text(dir / "optimizer.index.json", "{\"tensors\":[]}\n");
  return true;
}

bool load_transformer_artifact(const std::filesystem::path& dir,
                               TransformerState* state, std::string* error) {
  TransformerConfig cfg;
  if (!load_transformer_config(dir / "config.json", &cfg, error)) return false;
  init_transformer_state(cfg, state);
  auto index = read_text(dir / "weights.index.json");
  bool ok = true;
  params(*state, [&](const Parameter& p) {
    auto* mut = const_cast<Parameter*>(&p);
    ok = ok && read_param(dir, index, mut);
  });
  if (!ok) *error = "failed to load transformer artifact tensors";
  return ok;
}

}  // namespace lkjai
