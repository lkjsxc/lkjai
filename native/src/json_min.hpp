#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

std::string json_escape(std::string_view value);
std::string read_text(const std::filesystem::path& path);
bool contains_json_string(std::string_view text, std::string_view key,
                          std::string_view value);

}  // namespace lkjai
