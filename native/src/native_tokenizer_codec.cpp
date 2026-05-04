#include "native_tokenizer_internal.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace lkjai {
namespace {

std::unordered_map<unsigned char, std::string> byte_encoder() {
  std::vector<int> bs;
  for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
  for (int i = 0xa1; i <= 0xac; ++i) bs.push_back(i);
  for (int i = 0xae; i <= 0xff; ++i) bs.push_back(i);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n++);
    }
  }
  std::unordered_map<unsigned char, std::string> out;
  for (size_t i = 0; i < bs.size(); ++i) {
    out[static_cast<unsigned char>(bs[i])] = tokenizer_utf8_cp(cs[i]);
  }
  return out;
}

std::unordered_map<std::string, unsigned char> byte_decoder() {
  std::unordered_map<std::string, unsigned char> out;
  for (const auto& [byte, text] : byte_encoder()) out[text] = byte;
  return out;
}

}  // namespace

std::string tokenizer_utf8_cp(int cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
  return out;
}

std::vector<std::string> tokenizer_bpe(const NativeTokenizer& tokenizer,
                                       std::vector<std::string> pieces) {
  if (pieces.size() < 2) return pieces;
  while (true) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best = pieces.size();
    for (size_t i = 0; i + 1 < pieces.size(); ++i) {
      auto found = tokenizer.merge_ranks.find(pieces[i] + "\t" + pieces[i + 1]);
      if (found != tokenizer.merge_ranks.end() && found->second < best_rank) {
        best_rank = found->second;
        best = i;
      }
    }
    if (best == pieces.size()) break;
    pieces[best] += pieces[best + 1];
    pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best + 1));
  }
  return pieces;
}

std::vector<std::string> tokenizer_byte_pieces(std::string_view text) {
  static const auto enc = byte_encoder();
  std::vector<std::string> out;
  out.reserve(text.size());
  for (unsigned char ch : text) out.push_back(enc.at(ch));
  return out;
}

std::string tokenizer_decode_piece(const std::string& piece) {
  static const auto dec = byte_decoder();
  std::string out;
  for (size_t i = 0; i < piece.size();) {
    bool matched = false;
    for (int len : {3, 2, 1}) {
      if (i + static_cast<size_t>(len) > piece.size()) continue;
      auto found = dec.find(piece.substr(i, static_cast<size_t>(len)));
      if (found == dec.end()) continue;
      out.push_back(static_cast<char>(found->second));
      i += static_cast<size_t>(len);
      matched = true;
      break;
    }
    if (!matched) out.push_back(piece[i++]);
  }
  return out;
}

}  // namespace lkjai
