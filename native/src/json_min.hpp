#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace lkjai {

std::string json_escape(std::string_view value);
std::string read_text(const std::filesystem::path& path);
bool contains_json_string(std::string_view text, std::string_view key,
                          std::string_view value);
std::vector<std::string> json_string_values(std::string_view text,
                                            std::string_view key);
std::string json_first_string(std::string_view text, std::string_view key);
int json_int_value(std::string_view text, std::string_view key, int fallback);
bool json_bool_value(std::string_view text, std::string_view key, bool fallback);

}  // namespace lkjai
