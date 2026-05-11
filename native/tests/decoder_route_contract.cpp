#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "native_server_routes.hpp"
#include "transformer_state.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

std::string escape(std::string_view value) {
  std::string out;
  for (char ch : value) {
    if (ch == '"' || ch == '\\') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

std::filesystem::path write_tokenizer(const std::filesystem::path& dir) {
  auto path = dir / "tokenizer.json";
  std::ofstream out(path);
  out << "{\"model\":{\"type\":\"BPE\",\"vocab\":{";
  bool first = true;
  for (int ch = 33; ch <= 126; ++ch) {
    if (!first) out << ",";
    first = false;
    std::string token(1, static_cast<char>(ch));
    out << "\"" << escape(token) << "\":" << ch;
  }
  out << "}},\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"added_tokens\":[";
  int id = 256;
  for (const auto& tag : {"<pad>", "<unk>", "<bos>", "<eos>",
                          "<assistant_action>", "<dialogue>", "</dialogue>",
                          "<message>", "</message>", "<role>", "</role>",
                          "<tool_name>", "</tool_name>", "<content>",
                          "</content>", "<action>", "</action>"}) {
    if (id != 256) out << ",";
    out << "{\"id\":" << id++ << ",\"content\":\"" << tag
        << "\",\"special\":true}";
  }
  out << "],\"merges\":[]}\n";
  return path;
}

lkjai::TransformerConfig decoder_cfg() {
  lkjai::TransformerConfig cfg;
  cfg.model = "decoder-route";
  cfg.kind = "decoder";
  cfg.vocab_size = 512;
  cfg.context = 8;
  cfg.layers = 1;
  cfg.hidden_size = 32;
  cfg.heads = 4;
  cfg.kv_heads = 2;
  cfg.head_dim = 8;
  cfg.ffn_size = 64;
  cfg.tie_embeddings = true;
  return cfg;
}

bool decoder_route_contract() {
  auto root = std::filesystem::temp_directory_path() / "lkjai-decoder-route";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto tokenizer = write_tokenizer(root);
  lkjai::TransformerState state;
  init_transformer_state(decoder_cfg(), &state);
  std::string checksum;
  auto model_dir = root / "model";
  if (!write_transformer_artifact(model_dir, state, 1, 1, 1, 4, 1, 1.0,
                                  false, &checksum, tokenizer)) {
    return expect(false, "failed to write decoder artifact");
  }
  lkjai::ArtifactStatus artifact;
  artifact.loaded = true;
  artifact.model_name = "decoder-route";
  artifact.model_dir = model_dir;
  lkjai::CudaStatus cuda;
  lkjai::RuntimeConfig runtime{"127.0.0.1", 8080, root.string(),
                               "local-native-engine", artifact.model_name,
                               "readonly", root.string(), "", "default", ""};
  std::string body =
      "{\"model\":\"decoder-route\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}";
  auto resp = lkjai::native_server_route(
      {"POST", "/v1/chat/completions", body}, artifact, cuda, runtime);
  return expect(resp.status == 200, "decoder route status") &&
         expect(has(resp.body, "\"choices\""), "decoder choices present") &&
         expect(!has(resp.body, "\"lkjai_decode_backend\":\"cuda_kv_cache\""),
                "host decode must not claim CUDA KV-cache") &&
         expect(has(resp.body,
                    "\"lkjai_decode_backend\":\"host_reference_recompute\""),
                "partial decode backend") &&
         expect(has(resp.body,
                    "\"lkjai_kv_cache_backend\":\"host_contiguous_bf16_diagnostic\""),
                "partial kv backend") &&
         expect(has(resp.body, "\"lkjai_decode_supported\":false"),
                "partial decode disclosure") &&
         expect(has(resp.body, "\"lkjai_kv_steady_state_token_allocations\":0"),
                "zero steady-state allocations");
}

}  // namespace

int main() { return decoder_route_contract() ? 0 : 1; }
