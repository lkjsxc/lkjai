#include <iostream>
#include <string>

#include "json_min.hpp"
#include "native_tokenizer_build.hpp"

namespace {

std::string value(int argc, char** argv, const std::string& name) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return "";
}

}  // namespace

int main(int argc, char** argv) {
  auto out = value(argc, argv, "--out");
  if (out.empty()) out = "data/train/tokenizer/tokenizer.json";
  int max_vocab_size = 8192;
  auto max_vocab = value(argc, argv, "--max-vocab-size");
  if (!max_vocab.empty()) max_vocab_size = std::stoi(max_vocab);

  lkjai::NativeTokenizerBuildResult result;
  std::string error;
  if (!lkjai::build_native_tokenizer_json(out, max_vocab_size, &result,
                                          &error)) {
    std::cerr << "tokenizer build failed: " << error << "\n";
    return 2;
  }
  std::cout << "{\"status\":\"pass\",\"format\":\"tokenizer.json\","
            << "\"out\":\"" << lkjai::json_escape(out) << "\","
            << "\"vocab_size\":" << result.vocab_size << ","
            << "\"max_vocab_size\":" << max_vocab_size << ","
            << "\"tokenizer_digest\":\"" << result.digest << "\"}\n";
  return 0;
}
