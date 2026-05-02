#include <filesystem>
#include <iostream>
#include <string>

#include "dense_cuda.hpp"
#include "json_min.hpp"
#include "transformer_train.hpp"

namespace {

std::string value(int argc, char** argv, const std::string& name) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return "";
}

}  // namespace

int main(int argc, char** argv) {
  auto dir = value(argc, argv, "--model-dir");
  auto tokens = value(argc, argv, "--tokens");
  if (dir.empty() || tokens.empty()) {
    std::cerr << "usage: lkjai-native-logits-check --model-dir DIR --tokens CSV\n";
    return 2;
  }
  std::string json;
  std::string error;
  auto manifest = lkjai::read_text(std::filesystem::path(dir) / "manifest.json");
  bool ok = false;
  if (lkjai::contains_json_string(manifest, "kind", "dense")) {
    ok = lkjai::dense_cuda_logits_check(dir, tokens, &json, &error);
  } else {
    ok = lkjai::transformer_logits_check(dir, tokens, &json, &error);
  }
  if (!ok) {
    std::cerr << "native logits check failed: " << error << "\n";
    return 2;
  }
  std::cout << json << "\n";
  return 0;
}
