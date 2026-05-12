#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "native_server_routes.hpp"
#include "native_tokenizer_build.hpp"
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

std::filesystem::path write_tokenizer(const std::filesystem::path& dir) {
  auto path = dir / "tokenizer.json";
  lkjai::NativeTokenizerBuildResult result;
  std::string error;
  if (!lkjai::build_native_tokenizer_json(path, 512, &result, &error)) {
    std::cerr << error << "\n";
    std::exit(1);
  }
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
  bool ok = expect(resp.status == 200, "decoder route status") &&
            expect(has(resp.body, "\"choices\""), "decoder choices present") &&
            expect(!has(resp.body, "\"lkjai_decode_backend\":\"cuda_kv_cache\""),
                   "host decode must not claim CUDA KV-cache") &&
            expect(has(resp.body,
                       "\"lkjai_decode_backend\":\"cuda_reference_kv_cache\""),
                   "partial decode backend") &&
            expect(has(resp.body,
                       "\"lkjai_kv_cache_backend\":\"cuda_contiguous_bf16_partial\""),
                   "partial kv backend") &&
            expect(has(resp.body, "\"lkjai_decode_supported\":true"),
                   "partial decode disclosure") &&
            expect(has(resp.body, "\"lkjai_decode_accepted\":false"),
                   "non-accepted disclosure") &&
            expect(has(resp.body, "\"lkjai_kv_steady_state_token_allocations\":0"),
                   "zero steady-state allocations");
  std::ofstream(model_dir / "decoder_acceptance.json")
      << "{\"decode_supported\":true,\"decode_backend\":\"cuda_kv_cache\","
         "\"kv_cache_backend\":\"cuda_contiguous_bf16\"}\n";
  auto accepted = lkjai::native_server_route(
      {"POST", "/v1/chat/completions", body}, artifact, cuda, runtime);
  ok = ok && expect(accepted.status == 200, "sidecar route status") &&
         expect(!has(accepted.body,
                     "\"lkjai_decode_backend\":\"cuda_kv_cache\""),
                "sidecar must not promote incomplete CUDA KV-cache") &&
         expect(has(accepted.body,
                    "\"lkjai_decode_backend\":\"cuda_reference_kv_cache\""),
                "sidecar remains partial decode backend") &&
         expect(has(accepted.body, "\"lkjai_decode_accepted\":false"),
                "sidecar remains non-accepted disclosure");
  std::ofstream(model_dir / "decoder_acceptance.json")
      << "{\"decode_supported\":true,\"decode_backend\":\"cuda_kv_cache\","
         "\"kv_cache_backend\":\"cuda_contiguous_bf16\","
         "\"runtime_path\":\"accepted_cuda_kv_cache\","
         "\"kv_cache_steady_state_token_allocations\":0}\n";
  auto promoted = lkjai::native_server_route(
      {"POST", "/v1/chat/completions", body}, artifact, cuda, runtime);
  return ok && expect(promoted.status == 200, "accepted route status") &&
         expect(has(promoted.body,
                    "\"lkjai_decode_backend\":\"cuda_kv_cache\""),
                "accepted decode backend") &&
         expect(has(promoted.body,
                    "\"lkjai_kv_cache_backend\":\"cuda_contiguous_bf16\""),
                "accepted kv backend") &&
         expect(has(promoted.body, "\"lkjai_kv_prefill_allocated_bytes\":"),
                "accepted prefill bytes") &&
         expect(has(promoted.body,
                    "\"lkjai_kv_steady_state_token_allocations\":0"),
                "accepted zero steady-state allocations") &&
         expect(has(promoted.body, "\"lkjai_decode_accepted\":true"),
                "accepted disclosure");
}

}  // namespace

int main() { return decoder_route_contract() ? 0 : 1; }
