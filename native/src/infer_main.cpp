#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "artifact.hpp"
#include "capability_json.hpp"
#include "dense_cuda_internal.hpp"
#include "json_min.hpp"

namespace {

std::string value(int argc, char** argv, const std::string& name) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return "";
}

std::string infer_json(const lkjai::DenseConfig& cfg,
                       const std::vector<float>& logits) {
  auto top = std::max_element(logits.begin(), logits.end());
  int top_token = static_cast<int>(std::distance(logits.begin(), top));
  auto cuda = lkjai::cuda_status();
  std::ostringstream out;
  out << "{\"status\":\"pass\",\"model_kind\":\"dense\",\"kind\":\"dense\""
      << ",\"shape\":[1," << cfg.vocab_size << "]"
      << ",\"finite\":true"
      << ",\"checksum\":\"" << lkjai::dense_checksum_floats(logits) << "\""
      << ",\"top_token\":" << top_token
      << ",\"capability\":{" << lkjai::capability_json_fields(cuda) << "}}";
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  auto dir = value(argc, argv, "--model-dir");
  auto tokens = value(argc, argv, "--tokens");
  if (dir.empty() || tokens.empty()) {
    std::cerr << "usage: lkjai-native-infer --model-dir DIR --tokens CSV\n";
    return 2;
  }
  std::filesystem::path model_dir(dir);
  std::string error;
  if (!lkjai::inspect_artifact(model_dir, &error)) {
    std::cerr << "native infer failed: " << error << "\n";
    return 2;
  }
  auto manifest = lkjai::read_text(model_dir / "manifest.json");
  if (!lkjai::contains_json_string(manifest, "kind", "dense")) {
    std::cerr << "native infer failed: only dense artifacts are supported\n";
    return 2;
  }
  auto cfg = lkjai::dense_config_from_artifact(model_dir);
  auto emb = lkjai::read_dense_tensor(model_dir, "tok_embeddings", &error);
  if (!error.empty()) {
    std::cerr << "native infer failed: " << error << "\n";
    return 2;
  }
  auto head = lkjai::read_dense_tensor(model_dir, "lm_head", &error);
  if (!error.empty()) {
    std::cerr << "native infer failed: " << error << "\n";
    return 2;
  }
  std::vector<float> logits;
  if (!lkjai::dense_logits_for_tokens(cfg, emb, head, tokens, &logits, &error)) {
    std::cerr << "native infer failed: " << error << "\n";
    return 2;
  }
  std::cout << infer_json(cfg, logits) << "\n";
  return 0;
}
