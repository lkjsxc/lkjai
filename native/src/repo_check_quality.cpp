#include "repo_check.hpp"

#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

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
  if (path.filename() == "CMakeLists.txt") {
    *limit = 200;
    return true;
  }
  static const std::unordered_map<std::string, int> limits = {
      {".md", 300},  {".cpp", 200}, {".hpp", 200}, {".cu", 200},
      {".cuh", 200}, {".sh", 200},  {".toml", 200}, {".yml", 200},
      {".yaml", 200}, {".css", 200}, {".js", 200}, {".json", 200},
      {".cmake", 200}};
  auto found = limits.find(path.extension().string());
  if (found == limits.end()) return false;
  *limit = found->second;
  return true;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

bool text_file(const std::filesystem::path& path) {
  auto ext = path.extension().string();
  return ext == ".md" || ext == ".sh" || ext == ".yml" || ext == ".yaml" ||
         ext == ".cmake" || ext == ".json" || ext == ".txt" ||
         path.filename() == "CMakeLists.txt";
}

void check_native_only_file(const std::filesystem::path& file,
                            RepoCheckResult* result) {
  auto name = file.filename().string();
  auto ext = file.extension().string();
  if (ext == ".py" || ext == ".rs" || name == "Cargo.toml" ||
      name == "pyproject.toml" || name.rfind("requirements", 0) == 0) {
    result->fail("tracked Python/Rust product artifact is forbidden: " +
                 file.string());
  }
  if (!text_file(file)) return;
  auto body = read_file(file);
  static const std::vector<std::string> blocked = {
      std::string("python3 ") + "tools/",
      std::string("python ") + "tools/",
      std::string("python3 ") + "benchmarks/",
      std::string("python3 ") + "diagnostics/",
      std::string("python3 ") + "reports/",
      std::string("run_decoder_") + "2h.py",
      std::string("#!/usr/bin/env ") + "python",
      std::string("cargo ") + "build",
      std::string("cargo ") + "run",
      std::string("cargo ") + "test"};
  for (const auto& needle : blocked) {
    if (body.find(needle) != std::string::npos) {
      result->fail(file.string() + " contains forbidden workflow: " + needle);
    }
  }
}

}  // namespace

int check_line_limits(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_tracked_files(repo)) {
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
  for (const auto& file : collect_tracked_files(repo)) {
    if (file.filename() == "package.json") {
      result.fail("Node runtime manifest is forbidden: " + file.string());
    }
  }
  return result.errors == 0 ? 0 : 1;
}

int check_native_only(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_tracked_files(repo)) {
    check_native_only_file(file, &result);
  }
  return result.errors == 0 ? 0 : 1;
}

int check_stable_identifiers(const std::filesystem::path& repo) {
  RepoCheckResult result;
  static const std::vector<std::string> blocked = {
      std::string("lkjai-packed-cache-") + "v1",
      std::string("lkjai-packed-cache-") + "v2",
      std::string("lkjai-native-artifact-") + "v2",
      std::string("lkjai-agent-jsonl-") + "v2",
      std::string("lkjai-agent-jsonl-") + "v3",
      std::string("lkjai-train-config-") + "v1",
      std::string("kimi-sft-60m-") + "v2",
      std::string("pref-") + "v1",
      std::string("repo-grounding-") + "v1",
      std::string("schema_") + "version"};
  for (const auto& file : collect_tracked_files(repo)) {
    auto path = file.lexically_relative(repo).string();
    for (const auto& needle : blocked) {
      if (path.find(needle) != std::string::npos)
        result.fail(path + " has legacy identifier " + needle);
    }
    if (!text_file(file)) continue;
    auto body = read_file(file);
    for (const auto& needle : blocked) {
      if (body.find(needle) != std::string::npos)
        result.fail(path + " contains legacy identifier " + needle);
    }
  }
  return result.errors == 0 ? 0 : 1;
}

}  // namespace lkjai
