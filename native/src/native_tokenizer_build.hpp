#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

struct NativeTokenizerBuildResult {
  int vocab_size = 0;
  std::string digest;
};

bool build_native_tokenizer_json(const std::filesystem::path& out,
                                 int max_vocab_size,
                                 NativeTokenizerBuildResult* result,
                                 std::string* error);

}  // namespace lkjai
