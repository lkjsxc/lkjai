#include "native_tokenizer_internal.hpp"

#include <algorithm>
#include <cctype>

#include "json_min.hpp"

namespace lkjai {
namespace {

void skip_ws(std::string_view text, size_t* pos) {
  while (*pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*pos]))) ++*pos;
}

bool parse_string(std::string_view text, size_t* pos, std::string* out) {
  if (*pos >= text.size() || text[*pos] != '"') return false;
  out->clear();
  bool escaped = false;
  for (++*pos; *pos < text.size(); ++*pos) {
    char ch = text[*pos];
    if (!escaped && ch == '"') {
      ++*pos;
      return true;
    }
    if (!escaped && ch == '\\') {
      escaped = true;
      continue;
    }
    if (!escaped) {
      out->push_back(ch);
      continue;
    }
    if (ch == 'n') out->push_back('\n');
    else if (ch == 't') out->push_back('\t');
    else if (ch == 'u') {
      if (*pos + 4 >= text.size()) return false;
      int cp = 0;
      for (int i = 0; i < 4; ++i) {
        char h = text[++*pos];
        cp *= 16;
        if (h >= '0' && h <= '9') cp += h - '0';
        else if (h >= 'a' && h <= 'f') cp += 10 + h - 'a';
        else if (h >= 'A' && h <= 'F') cp += 10 + h - 'A';
        else return false;
      }
      *out += tokenizer_utf8_cp(cp);
    } else if (ch != 'r') {
      out->push_back(ch);
    }
    escaped = false;
  }
  return false;
}

bool parse_int_at(std::string_view text, size_t* pos, int* out) {
  skip_ws(text, pos);
  try {
    size_t used = 0;
    *out = std::stoi(std::string(text.substr(*pos)), &used);
    *pos += used;
    return true;
  } catch (...) {
    return false;
  }
}

std::string object_slice(std::string_view text, std::string_view key) {
  auto pos = text.find("\"" + std::string(key) + "\"");
  if (pos == std::string_view::npos) return "";
  pos = text.find('{', pos);
  if (pos == std::string_view::npos) return "";
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = pos; i < text.size(); ++i) {
    char ch = text[i];
    if (in_string) {
      if (escaped) escaped = false;
      else if (ch == '\\') escaped = true;
      else if (ch == '"') in_string = false;
    } else if (ch == '"') {
      in_string = true;
    } else if (ch == '{') {
      ++depth;
    } else if (ch == '}' && --depth == 0) {
      return std::string(text.substr(pos, i - pos + 1));
    }
  }
  return "";
}

bool parse_vocab(std::string_view text, NativeTokenizer* t) {
  auto vocab = object_slice(text, "vocab");
  if (vocab.empty()) return false;
  size_t pos = 1;
  while (pos < vocab.size()) {
    skip_ws(vocab, &pos);
    if (pos < vocab.size() && vocab[pos] == '}') break;
    std::string token;
    int id = 0;
    if (!parse_string(vocab, &pos, &token)) return false;
    skip_ws(vocab, &pos);
    if (pos >= vocab.size() || vocab[pos++] != ':') return false;
    if (!parse_int_at(vocab, &pos, &id)) return false;
    t->vocab[token] = id;
    t->id_to_token[id] = token;
    t->vocab_size = std::max(t->vocab_size, id + 1);
    skip_ws(vocab, &pos);
    if (pos < vocab.size() && vocab[pos] == ',') ++pos;
  }
  return true;
}

bool parse_added(std::string_view text, NativeTokenizer* t) {
  auto pos = text.find("\"added_tokens\"");
  if (pos == std::string_view::npos) return true;
  pos = text.find('[', pos);
  if (pos == std::string_view::npos) return false;
  while ((pos = text.find('{', pos)) != std::string_view::npos) {
    auto end = text.find('}', pos);
    if (end == std::string_view::npos) return false;
    auto item = text.substr(pos, end - pos + 1);
    int id = json_int_value(item, "id", -1);
    auto content = json_first_string(item, "content");
    if (id >= 0 && !content.empty()) {
      t->added_tokens[content] = id;
      t->added_id_to_token[id] = content;
      t->id_to_token[id] = content;
      t->vocab_size = std::max(t->vocab_size, id + 1);
      auto special = item.find("\"special\"");
      auto is_special = special != std::string_view::npos &&
                        item.find("true", special) != std::string_view::npos;
      if (is_special)
        t->special_ids[id] = true;
    }
    pos = end + 1;
    skip_ws(text, &pos);
    if (pos < text.size() && text[pos] == ']') break;
  }
  return true;
}

bool parse_merges(std::string_view text, NativeTokenizer* t) {
  auto pos = text.find("\"merges\"");
  if (pos == std::string_view::npos) return true;
  pos = text.find('[', pos);
  if (pos == std::string_view::npos) return false;
  int rank = 0;
  while ((pos = text.find('[', pos + 1)) != std::string_view::npos) {
    std::string a, b;
    ++pos;
    skip_ws(text, &pos);
    if (!parse_string(text, &pos, &a)) return false;
    pos = text.find(',', pos);
    if (pos == std::string_view::npos) return false;
    ++pos;
    skip_ws(text, &pos);
    if (!parse_string(text, &pos, &b)) return false;
    t->merge_ranks[a + "\t" + b] = rank++;
    pos = text.find(']', pos);
    if (pos == std::string_view::npos) return false;
    if (pos + 1 < text.size() && text[pos + 1] == ']') break;
  }
  return true;
}

}  // namespace

bool parse_tokenizer_json(std::string_view text, NativeTokenizer* tokenizer) {
  if (!parse_vocab(text, tokenizer) || !parse_added(text, tokenizer) ||
      !parse_merges(text, tokenizer)) return false;
  tokenizer->unk_id = tokenizer_id(*tokenizer, "<unk>", tokenizer->unk_id);
  tokenizer->eos_id = tokenizer_id(*tokenizer, "<eos>", tokenizer->eos_id);
  return true;
}

}  // namespace lkjai
