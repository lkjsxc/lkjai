#include <iostream>
#include <string>

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
  if (!lkjai::transformer_logits_check(dir, tokens, &json, &error)) {
    std::cerr << "native logits check failed: " << error << "\n";
    return 2;
  }
  std::cout << json << "\n";
  return 0;
}
