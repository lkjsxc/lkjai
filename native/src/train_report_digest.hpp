#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

std::string train_report_file_digest(const std::filesystem::path& path);
std::string train_report_packed_cache_digest(const std::filesystem::path& dir);
std::string train_report_manifest_checksum(const std::filesystem::path& dir);

}  // namespace lkjai
