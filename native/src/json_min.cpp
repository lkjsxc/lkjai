#include "json_min.hpp"

#include <fstream>
#include <sstream>

namespace lkjai {

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
  auto quoted_key = "\"" + std::string(key) + "\"";
  auto found = text.find(quoted_key);
  if (found == std::string_view::npos) {
    return false;
  }
  auto quoted_value = "\"" + std::string(value) + "\"";
  return text.find(quoted_value, found) != std::string_view::npos;
}

}  // namespace lkjai
