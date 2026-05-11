#include "json_min.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace lkjai {
namespace {

std::string quoted(std::string_view value) {
  return "\"" + std::string(value) + "\"";
}

size_t value_start(std::string_view text, size_t pos) {
  pos = text.find(':', pos);
  if (pos == std::string_view::npos) return pos;
  ++pos;
  while (pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  return pos;
}

}  // namespace

std::string json_escape(std::string_view value) {
  std::string out;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        break;
      default:
        out += ch;
    }
  }
  return out;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

bool contains_json_string(std::string_view text, std::string_view key,
                          std::string_view value) {
  auto quoted_key = quoted(key);
  auto found = text.find(quoted_key);
  if (found == std::string_view::npos) {
    return false;
  }
  auto quoted_value = quoted(value);
  return text.find(quoted_value, found) != std::string_view::npos;
}

std::vector<std::string> json_string_values(std::string_view text,
                                            std::string_view key) {
  std::vector<std::string> values;
  const auto needle = quoted(key);
  size_t search = 0;
  while (true) {
    auto pos = text.find(needle, search);
    if (pos == std::string_view::npos) break;
    pos = value_start(text, pos + needle.size());
    if (pos == std::string_view::npos || pos >= text.size() || text[pos] != '"') {
      search = pos == std::string_view::npos ? text.size() : pos + 1;
      continue;
    }
    std::string out;
    bool escaped = false;
    for (size_t i = pos + 1; i < text.size(); ++i) {
      search = i + 1;
      char ch = text[i];
      if (escaped) {
        if (ch == 'n') out.push_back('\n');
        else if (ch == 't') out.push_back('\t');
        else if (ch != 'r') out.push_back(ch);
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        values.push_back(out);
        break;
      } else {
        out.push_back(ch);
      }
    }
  }
  return values;
}

std::string json_first_string(std::string_view text, std::string_view key) {
  auto values = json_string_values(text, key);
  return values.empty() ? "" : values.front();
}

int json_int_value(std::string_view text, std::string_view key, int fallback) {
  const auto needle = quoted(key);
  auto pos = text.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = value_start(text, pos + needle.size());
  if (pos == std::string_view::npos || pos >= text.size()) return fallback;
  try {
    return std::stoi(std::string(text.substr(pos)));
  } catch (...) {
    return fallback;
  }
}

double json_double_value(std::string_view text, std::string_view key,
                         double fallback) {
  const auto needle = quoted(key);
  auto pos = text.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = value_start(text, pos + needle.size());
  if (pos == std::string_view::npos || pos >= text.size()) return fallback;
  try {
    return std::stod(std::string(text.substr(pos)));
  } catch (...) {
    return fallback;
  }
}

bool json_bool_value(std::string_view text, std::string_view key, bool fallback) {
  const auto needle = quoted(key);
  auto pos = text.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = value_start(text, pos + needle.size());
  if (pos == std::string_view::npos || pos >= text.size()) return fallback;
  if (text.substr(pos, 4) == "true") return true;
  if (text.substr(pos, 5) == "false") return false;
  return fallback;
}

}  // namespace lkjai
