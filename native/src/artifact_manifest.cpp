#include "artifact_manifest.hpp"

#include "artifact.hpp"
#include "json_min.hpp"

#include <cstdint>
#include <sstream>

namespace lkjai {

std::string artifact_text_checksum(std::string_view text) {
  uint64_t hash = 1469598103934665603ull;
  for (char ch : text) {
    hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

bool validate_manifest(std::string_view text, std::string_view config,
                       std::string_view tokenizer, std::string* kind,
                       std::string* error) {
  if (!contains_json_string(text, "format", "lkjai-native-artifact")) {
    *error = "manifest format must be lkjai-native-artifact";
    return false;
  }
  *kind = json_first_string(text, "artifact_kind");
  if (*kind != "export" && *kind != "checkpoint") {
    *error = "manifest artifact_kind must be export or checkpoint";
    return false;
  }
  if (!contains_json_string(text, "config_checksum",
                            artifact_text_checksum(config))) {
    *error = "manifest config_checksum mismatch";
    return false;
  }
  if (!contains_json_string(text, "tokenizer_checksum",
                            artifact_text_checksum(tokenizer))) {
    *error = "manifest tokenizer_checksum mismatch";
    return false;
  }
  return true;
}

}  // namespace lkjai
