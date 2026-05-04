#pragma once

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lkjai {

struct NativeTokenizer {
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<int, std::string> id_to_token;
  std::unordered_map<std::string, int> added_tokens;
  std::unordered_map<int, std::string> added_id_to_token;
  std::unordered_map<int, bool> special_ids;
  std::unordered_map<std::string, int> merge_ranks;
  int unk_id = 1;
  int eos_id = 3;
  int vocab_size = 0;
};

bool load_native_tokenizer(const std::filesystem::path& path,
                           NativeTokenizer* tokenizer,
                           std::string* error);
bool validate_decoder_tokenizer(const NativeTokenizer& tokenizer, int config_vocab,
                                std::string* error);
bool serialize_chat_prompt(std::string_view request_body, std::string* prompt,
                           std::string* error);
std::vector<uint16_t> tokenizer_encode(const NativeTokenizer& tokenizer,
                                       std::string_view text);
std::string tokenizer_decode(const NativeTokenizer& tokenizer,
                             const std::vector<uint16_t>& ids,
                             bool skip_special_tokens);
int tokenizer_id(const NativeTokenizer& tokenizer, const std::string& token,
                 int fallback);
int sample_next_token(const std::vector<float>& logits, float temperature,
                      int top_k, float top_p, uint64_t seed, int step);

}  // namespace lkjai
