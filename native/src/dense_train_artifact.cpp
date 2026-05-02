#include "dense_train_internal.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <sstream>

#include "artifact.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

uint16_t bf16(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

std::string hex64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << text;
}

void pad_256(std::ofstream& out) {
  auto pos = static_cast<uint64_t>(out.tellp());
  for (uint64_t i = pos % 256; i != 0 && i < 256; ++i) out.put('\0');
}

void write_bf16(std::ofstream& out, const std::vector<float>& values) {
  for (float value : values) {
    auto packed = bf16(value);
    out.write(reinterpret_cast<const char*>(&packed), sizeof(packed));
  }
}

void append_named(std::ofstream& weights, std::ostringstream* index,
                  const std::string& name, const std::vector<int>& shape,
                  const std::vector<float>& values, bool* first,
                  uint64_t* hash) {
  pad_256(weights);
  uint64_t offset = static_cast<uint64_t>(weights.tellp());
  write_bf16(weights, values);
  uint64_t bytes = static_cast<uint64_t>(weights.tellp()) - offset;
  if (!*first) *index << ",";
  *first = false;
  *index << "{\"name\":\"" << json_escape(name) << "\",\"dtype\":\"bf16\",";
  *index << "\"shape\":[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) *index << ",";
    *index << shape[i];
  }
  *index << "],\"byte_offset\":" << offset << ",\"byte_length\":" << bytes
         << "}";
  for (float value : values) *hash = (*hash ^ bf16(value)) * 1099511628211ull;
}

void append_f32(std::ofstream& weights, std::ostringstream* index,
                const std::string& name, const std::vector<int>& shape,
                const std::vector<float>& values, bool* first) {
  pad_256(weights);
  uint64_t offset = static_cast<uint64_t>(weights.tellp());
  weights.write(reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
  uint64_t bytes = static_cast<uint64_t>(weights.tellp()) - offset;
  if (!*first) *index << ",";
  *first = false;
  *index << "{\"name\":\"" << json_escape(name) << "\",\"dtype\":\"f32\",";
  *index << "\"shape\":[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) *index << ",";
    *index << shape[i];
  }
  *index << "],\"byte_offset\":" << offset << ",\"byte_length\":" << bytes
         << "}";
}

std::string config_json(const DenseConfig& c) {
  std::ostringstream out;
  out << "{\"model\":\"" << json_escape(c.model) << "\",\"dtype\":\"bf16\","
      << "\"vocab_size\":" << c.vocab_size << ",\"context\":" << c.context
      << ",\"layers\":" << c.layers << ",\"hidden_size\":" << c.hidden_size
      << ",\"heads\":" << c.heads << ",\"kv_heads\":" << c.kv_heads
      << ",\"head_dim\":" << c.head_dim << ",\"ffn_size\":" << c.ffn_size
      << ",\"seed\":" << c.seed << "}\n";
  return out.str();
}

std::string manifest_json(const std::string& artifact_kind,
                          std::string_view config,
                          std::string_view tokenizer) {
  std::ostringstream out;
  out << "{\"format\":\"lkjai-native-artifact-v2\",\"kind\":\"dense\","
      << "\"artifact_kind\":\"" << artifact_kind << "\","
      << "\"config_checksum\":\"" << artifact_text_checksum(config) << "\","
      << "\"tokenizer_checksum\":\"" << artifact_text_checksum(tokenizer)
      << "\"}\n";
  return out.str();
}

}  // namespace

bool write_dense_train_artifact(const std::filesystem::path& dir,
                                const DenseTrainState& state, int step,
                                double loss, bool checkpoint,
                                std::string* checksum) {
  std::filesystem::create_directories(dir);
  std::ofstream weights(dir / "weights.lkjw", std::ios::binary);
  if (!weights) return false;
  const auto& c = state.cfg;
  std::ostringstream index;
  bool first = true;
  uint64_t hash = 1469598103934665603ull;
  index << "{\"tensors\":[";
  append_named(weights, &index, "tok_embeddings",
               {c.vocab_size, c.hidden_size}, state.emb, &first, &hash);
  append_named(weights, &index, "lm_head", {c.vocab_size, c.hidden_size},
               state.head, &first, &hash);
  index << "]}\n";
  weights.close();
  *checksum = hex64(hash);
  write_text(dir / "weights.index.json", index.str());
  auto config = config_json(c);
  auto tokenizer = "{\"format\":\"uint16-packed-cache\",\"vocab_size\":" +
                   std::to_string(c.vocab_size) + "}\n";
  write_text(dir / "manifest.json",
             manifest_json(checkpoint ? "checkpoint" : "export", config,
                           tokenizer));
  write_text(dir / "config.json", config);
  write_text(dir / "tokenizer.json", tokenizer);
  write_text(dir / "trainer_state.json",
             "{\"optimizer_steps\":" + std::to_string(step) +
                 ",\"loss\":" + std::to_string(loss) +
                 ",\"logits_checksum\":\"" + *checksum +
                 "\",\"checkpoint\":" + (checkpoint ? "true" : "false") +
                 "}\n");
  if (!checkpoint) return true;
  std::ofstream opt_file(dir / "optimizer.lkjw", std::ios::binary);
  if (!opt_file) return false;
  std::ostringstream opt_index;
  bool opt_first = true;
  opt_index << "{\"tensors\":[";
  append_f32(opt_file, &opt_index, "master.tok_embeddings",
             {c.vocab_size, c.hidden_size}, state.emb, &opt_first);
  append_f32(opt_file, &opt_index, "adam_m.tok_embeddings",
             {c.vocab_size, c.hidden_size}, state.m_emb, &opt_first);
  append_f32(opt_file, &opt_index, "adam_v.tok_embeddings",
             {c.vocab_size, c.hidden_size}, state.v_emb, &opt_first);
  append_f32(opt_file, &opt_index, "master.lm_head",
             {c.vocab_size, c.hidden_size}, state.head, &opt_first);
  append_f32(opt_file, &opt_index, "adam_m.lm_head",
             {c.vocab_size, c.hidden_size}, state.m_head, &opt_first);
  append_f32(opt_file, &opt_index, "adam_v.lm_head",
             {c.vocab_size, c.hidden_size}, state.v_head, &opt_first);
  opt_index << "]}\n";
  write_text(dir / "optimizer.index.json", opt_index.str());
  return true;
}

}  // namespace lkjai
