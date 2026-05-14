#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace lkjai_test {
namespace {

class JsonCursor {
 public:
  explicit JsonCursor(std::string_view input) : text_(input) {}

  bool parse() {
    skip_ws();
    if (!value()) return false;
    skip_ws();
    return pos_ == text_.size();
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool take(char ch) {
    skip_ws();
    if (pos_ >= text_.size() || text_[pos_] != ch) return false;
    ++pos_;
    return true;
  }

  bool literal(std::string_view token) {
    skip_ws();
    if (text_.substr(pos_, token.size()) != token) return false;
    pos_ += token.size();
    return true;
  }

  bool string_value() {
    skip_ws();
    if (pos_ >= text_.size() || text_[pos_] != '"') return false;
    ++pos_;
    while (pos_ < text_.size()) {
      char ch = text_[pos_++];
      if (ch == '"') return true;
      if (static_cast<unsigned char>(ch) < 0x20) return false;
      if (ch != '\\') continue;
      if (pos_ >= text_.size()) return false;
      char esc = text_[pos_++];
      if (std::string_view("\"\\/bfnrt").find(esc) != std::string_view::npos) {
        continue;
      }
      if (esc != 'u') return false;
      for (int n = 0; n < 4; ++n) {
        if (pos_ >= text_.size() ||
            !std::isxdigit(static_cast<unsigned char>(text_[pos_++]))) {
          return false;
        }
      }
    }
    return false;
  }

  bool number() {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    if (pos_ >= text_.size()) return false;
    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      if (!std::isdigit(static_cast<unsigned char>(text_[pos_]))) return false;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        return false;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        return false;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    return true;
  }

  bool array() {
    if (!take('[')) return false;
    skip_ws();
    if (take(']')) return true;
    do {
      if (!value()) return false;
    } while (take(','));
    return take(']');
  }

  bool object() {
    if (!take('{')) return false;
    skip_ws();
    if (take('}')) return true;
    do {
      if (!string_value() || !take(':') || !value()) return false;
    } while (take(','));
    return take('}');
  }

  bool value() {
    skip_ws();
    if (pos_ >= text_.size()) return false;
    char ch = text_[pos_];
    if (ch == '{') return object();
    if (ch == '[') return array();
    if (ch == '"') return string_value();
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return number();
    }
    return literal("true") || literal("false") || literal("null");
  }

  std::string_view text_;
  size_t pos_ = 0;
};

bool valid_json(const std::string& text) {
  return JsonCursor(text).parse();
}

}  // namespace
}  // namespace lkjai_test
