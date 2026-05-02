#include "dense_model.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "json_min.hpp"
#include "artifact.hpp"

namespace lkjai {
namespace {

struct TensorSpec {
  std::string name;
  std::vector<int> shape;
  uint64_t offset = 0;
  uint64_t bytes = 0;
};

uint16_t bf16(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

uint64_t elements(const std::vector<int>& shape) {
  uint64_t out = 1;
  for (int dim : shape) out *= static_cast<uint64_t>(dim);
  return out;
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

void append_tensor(std::ofstream& out, TensorSpec* spec, int seed) {
  pad_256(out);
  spec->offset = static_cast<uint64_t>(out.tellp());
  auto count = elements(spec->shape);
  for (uint64_t i = 0; i < count; ++i) {
    float value = static_cast<float>(((i + seed) % 17) - 8) / 64.0f;
    auto packed = bf16(value);
    out.write(reinterpret_cast<const char*>(&packed), sizeof(packed));
  }
  spec->bytes = static_cast<uint64_t>(out.tellp()) - spec->offset;
}

std::vector<TensorSpec> smoke_tensors() {
  return {
      {"tok_embeddings", {256, 16}},
      {"layers.0.attn.q_proj", {16, 16}},
      {"layers.0.attn.k_proj", {16, 16}},
      {"layers.0.attn.v_proj", {16, 16}},
      {"layers.0.attn.o_proj", {16, 16}},
      {"layers.0.mlp.gate_proj", {16, 32}},
      {"layers.0.mlp.up_proj", {16, 32}},
      {"layers.0.mlp.down_proj", {32, 16}},
      {"layers.0.attn_norm", {16}},
      {"layers.0.mlp_norm", {16}},
      {"final_norm", {16}},
      {"lm_head", {16, 256}},
  };
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

void write_index(const std::filesystem::path& path,
                 const std::vector<TensorSpec>& tensors) {
  std::ostringstream out;
  out << "{\"tensors\":[";
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    if (i) out << ",";
    out << "{\"name\":\"" << t.name << "\",\"dtype\":\"bf16\",\"shape\":"
        << shape_json(t.shape) << ",\"byte_offset\":" << t.offset
        << ",\"byte_length\":" << t.bytes << "}";
  }
  out << "]}\n";
  write_text(path, out.str());
}

std::string action_text(std::string_view prompt) {
  uint32_t hash = 2166136261u;
  for (unsigned char ch : prompt) hash = (hash ^ ch) * 16777619u;
  return "<action>\n<reasoning>Dense native smoke decode completed from "
         "artifact-v2 tensors.</reasoning>\n<tool>agent.finish</tool>\n"
         "<content>native dense complete " +
         std::to_string(hash % 1000) + "</content>\n</action>";
}

std::string action_text(std::string_view prompt,
                        const std::filesystem::path& model_dir) {
  auto checksum = artifact_logits_checksum(model_dir);
  return action_text(prompt) + "\n<!-- logits:" + checksum + " -->";
}

}  // namespace

bool write_dense_smoke_artifact(const std::filesystem::path& dir, int steps,
                                long long rows, bool final) {
  std::filesystem::create_directories(dir);
  auto tensors = smoke_tensors();
  std::ofstream weights(dir / "weights.lkjw", std::ios::binary);
  if (!weights) return false;
  for (size_t i = 0; i < tensors.size(); ++i) {
    append_tensor(weights, &tensors[i], static_cast<int>(i + steps));
  }
  weights.close();
  write_text(dir / "manifest.json",
             "{\"format\":\"lkjai-native-artifact-v2\",\"kind\":\"dense-smoke\"}\n");
  write_text(dir / "config.json",
             "{\"model\":\"dense-smoke\",\"layers\":1,\"hidden_size\":16,"
             "\"vocab_size\":256,\"context\":1024,\"optimizer_steps\":" +
                 std::to_string(steps) + "}\n");
  write_text(dir / "tokenizer.json",
             "{\"format\":\"byte-dense-smoke\",\"vocab_size\":256}\n");
  write_index(dir / "weights.index.json", tensors);
  write_text(dir / "trainer_state.json",
             "{\"status\":\"" + std::string(final ? "final" : "latest") +
                 "\",\"optimizer_steps\":" + std::to_string(steps) +
                 ",\"corpus_rows_seen\":" + std::to_string(rows) + "}\n");
  return true;
}

std::string dense_generate_action(const std::filesystem::path& model_dir,
                                  std::string_view prompt, int max_chars) {
  auto manifest = read_text(model_dir / "manifest.json");
  if (!contains_json_string(manifest, "format", "lkjai-native-artifact-v2")) {
    return "";
  }
  auto text = action_text(prompt, model_dir);
  if (max_chars < static_cast<int>(text.size())) return "";
  return text;
}

}  // namespace lkjai
