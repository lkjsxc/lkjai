#include "repo_check.hpp"

#include <fstream>
#include <string>
#include <unordered_map>

namespace lkjai {
namespace {

int count_lines(const std::filesystem::path& path) {
  std::ifstream file(path);
  int lines = 0;
  std::string line;
  while (std::getline(file, line)) ++lines;
  return lines;
}

bool limited_extension(const std::filesystem::path& path, int* limit) {
  static const std::unordered_map<std::string, int> limits = {
      {".md", 300},  {".cpp", 200}, {".hpp", 200}, {".cu", 200},
      {".cuh", 200}, {".sh", 200},  {".toml", 200}, {".yml", 200},
      {".yaml", 200}, {".css", 200}, {".js", 200}, {".json", 200}};
  auto found = limits.find(path.extension().string());
  if (found == limits.end()) return false;
  *limit = found->second;
  return true;
}

}  // namespace

int check_line_limits(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_files(repo)) {
    int limit = 0;
    if (!limited_extension(file, &limit)) continue;
    int lines = count_lines(file);
    if (lines > limit) {
      result.fail(file.string() + " has " + std::to_string(lines) +
                  " lines; limit is " + std::to_string(limit));
    }
  }
  return result.errors == 0 ? 0 : 1;
}

int check_no_node(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_files(repo)) {
    if (file.filename() == "package.json") {
      result.fail("Node runtime manifest is forbidden: " + file.string());
    }
  }
  return result.errors == 0 ? 0 : 1;
}

}  // namespace lkjai
