#include "native_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "json_min.hpp"
#include "native_tokenizer_internal.hpp"
#include "native_xml_tags.hpp"

namespace lkjai {

bool load_native_tokenizer(const std::filesystem::path& path,
                           NativeTokenizer* tokenizer,
                           std::string* error) {
  auto text = read_text(path);
  if (text.empty()) {
    *error = "empty or missing tokenizer: " + path.string();
    return false;
  }
  if (!contains_json_string(text, "type", "BPE") ||
      !contains_json_string(text, "type", "ByteLevel")) {
    *error = "tokenizer must be the repo byte-level BPE tokenizer subset";
    return false;
  }
  NativeTokenizer parsed;
  if (!parse_tokenizer_json(text, &parsed)) {
    *error = "failed to parse tokenizer.json";
    return false;
  }
  *tokenizer = std::move(parsed);
  return true;
}

int tokenizer_id(const NativeTokenizer& tokenizer, const std::string& token,
                 int fallback) {
  if (auto it = tokenizer.added_tokens.find(token);
      it != tokenizer.added_tokens.end()) return it->second;
  if (auto it = tokenizer.vocab.find(token); it != tokenizer.vocab.end())
    return it->second;
  return fallback;
}

bool validate_decoder_tokenizer(const NativeTokenizer& tokenizer, int config_vocab,
                                std::string* error) {
  if (tokenizer.vocab_size <= 0 || tokenizer.vocab_size > config_vocab) {
    *error = "tokenizer vocab_size exceeds decoder config vocab_size";
    return false;
  }
  for (const auto& tag : kNativeXmlTags) {
    if (tokenizer_id(tokenizer, tag.text, -1) < 0) {
      *error = std::string("tokenizer missing atomic tag ") + tag.text;
      return false;
    }
  }
  return true;
}

std::vector<uint16_t> tokenizer_encode(const NativeTokenizer& tokenizer,
                                       std::string_view text) {
  std::vector<uint16_t> out;
  for (size_t pos = 0; pos < text.size();) {
    int best_id = -1;
    size_t best_len = 0;
    for (const auto& [token, id] : tokenizer.added_tokens) {
      if (token.size() > best_len &&
          text.substr(pos, token.size()) == token) {
        best_id = id;
        best_len = token.size();
      }
    }
    if (best_id >= 0) {
      out.push_back(static_cast<uint16_t>(best_id));
      pos += best_len;
      continue;
    }
    size_t next = text.find('<', pos + 1);
    auto chunk = text.substr(pos, next == std::string_view::npos
                                      ? text.size() - pos
                                      : next - pos);
    for (const auto& piece :
         tokenizer_bpe(tokenizer, tokenizer_byte_pieces(chunk))) {
      out.push_back(static_cast<uint16_t>(
          tokenizer_id(tokenizer, piece, tokenizer.unk_id)));
    }
    pos += chunk.size();
  }
  if (out.empty()) out.push_back(static_cast<uint16_t>(tokenizer.unk_id));
  return out;
}

std::string tokenizer_decode(const NativeTokenizer& tokenizer,
                             const std::vector<uint16_t>& ids,
                             bool skip_special_tokens) {
  std::string out;
  for (auto id16 : ids) {
    int id = static_cast<int>(id16);
    if (skip_special_tokens && tokenizer.special_ids.contains(id)) continue;
    if (auto added = tokenizer.added_id_to_token.find(id);
        added != tokenizer.added_id_to_token.end()) {
      out += added->second;
      continue;
    }
    auto found = tokenizer.id_to_token.find(id);
    if (found != tokenizer.id_to_token.end())
      out += tokenizer_decode_piece(found->second);
  }
  return out;
}

int sample_next_token(const std::vector<float>& logits, float temperature,
                      int top_k, float top_p, uint64_t seed, int step) {
  if (logits.empty()) return 0;
  if (temperature <= 0.0f) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }
  std::vector<std::pair<float, int>> rows;
  rows.reserve(logits.size());
  for (int i = 0; i < static_cast<int>(logits.size()); ++i)
    rows.push_back({logits[static_cast<size_t>(i)] / temperature, i});
  std::sort(rows.begin(), rows.end(),
            [](auto a, auto b) { return a.first > b.first; });
  if (top_k > 0 && top_k < static_cast<int>(rows.size()))
    rows.resize(static_cast<size_t>(top_k));
  float max_logit = rows.front().first;
  double total = 0.0;
  for (auto& row : rows) {
    row.first = std::exp(row.first - max_logit);
    total += row.first;
  }
  if (top_p > 0.0f && top_p < 1.0f) {
    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < rows.size(); ++keep) {
      cumulative += rows[keep].first / total;
      if (cumulative >= top_p) break;
    }
    rows.resize(std::max<size_t>(1, keep + 1));
    total = 0.0;
    for (const auto& row : rows) total += row.first;
  }
  std::mt19937_64 rng(seed + static_cast<uint64_t>(step) * 0x9e3779b97f4a7c15ull);
  std::uniform_real_distribution<double> dist(0.0, total);
  double pick = dist(rng);
  for (const auto& row : rows) {
    pick -= row.first;
    if (pick <= 0.0) return row.second;
  }
  return rows.back().second;
}

}  // namespace lkjai
