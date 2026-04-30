#include "env.hpp"

#include <cstdlib>

namespace lkjai {

std::string env_string(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || value[0] == '\0' ? fallback : value;
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

}  // namespace lkjai
