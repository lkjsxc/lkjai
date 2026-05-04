#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "native_tokenizer.hpp"

namespace lkjai {

std::string tokenizer_utf8_cp(int cp);
bool parse_tokenizer_json(std::string_view text, NativeTokenizer* tokenizer);
std::vector<std::string> tokenizer_bpe(const NativeTokenizer& tokenizer,
                                       std::vector<std::string> pieces);
std::vector<std::string> tokenizer_byte_pieces(std::string_view text);
std::string tokenizer_decode_piece(const std::string& piece);

}  // namespace lkjai
