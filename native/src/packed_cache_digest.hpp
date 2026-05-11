#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

std::string packed_file_digest(const std::filesystem::path& path);
std::string packed_source_digest(const std::filesystem::path& path);
std::string packed_payload_digest(const std::filesystem::path& dir);

}  // namespace lkjai
