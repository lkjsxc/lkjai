#pragma once

#include <string>

namespace lkjai {

std::string env_string(const char* name, const std::string& fallback);
int env_int(const char* name, int fallback);

}  // namespace lkjai
