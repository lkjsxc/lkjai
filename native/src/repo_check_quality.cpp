#include "repo_check.hpp"

#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
      {".cmake", 200}, {".py", 200}};
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
         ext == ".cmake" || ext == ".json" || ext == ".txt" || ext == ".py" ||
         ext == ".cpp" || ext == ".hpp" || ext == ".cu" || ext == ".cuh" ||
         path.filename() == "CMakeLists.txt";
}

bool corpus_python_file(const std::filesystem::path& repo,
                        const std::filesystem::path& file) {
  auto rel = file.lexically_relative(repo).string();
  return rel.rfind("ops/corpus/", 0) == 0;
}

std::string relative_string(const std::filesystem::path& repo,
                            const std::filesystem::path& file) {
  return file.lexically_relative(repo).string();
}

bool durable_dir(const std::string& rel) {
  return !rel.empty() && rel.rfind("data", 0) != 0 && rel.rfind("artifacts", 0) != 0;
}

void check_readme_child_mentions(const std::filesystem::path& repo,
    const std::filesystem::path& dir, const std::unordered_set<std::string>& dirs,
    RepoCheckResult* result) {
  auto readme = dir / "README.md";
  if (!std::filesystem::is_regular_file(readme)) {
    result->fail("missing durable directory README: " + dir.string());
    return;
  }
  auto body = read_file(readme);
  auto rel = relative_string(repo, dir);
  for (const auto& child : dirs) {
    auto parent = std::filesystem::path(child).parent_path().string();
    if (parent != rel) continue;
    auto name = std::filesystem::path(child).filename().string();
    if (body.find(name) == std::string::npos) {
      result->fail(readme.string() + " missing child directory " + name);
    }
  }
}

void check_native_only_file(const std::filesystem::path& repo,
                            const std::filesystem::path& file,
                            RepoCheckResult* result) {
  auto name = file.filename().string();
  auto ext = file.extension().string();
  if ((ext == ".py" && !corpus_python_file(repo, file)) || ext == ".rs" ||
      name == "Cargo.toml" ||
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

int check_repo_readmes(const std::filesystem::path& repo) {
  RepoCheckResult result;
  std::unordered_set<std::string> dirs;
  for (const auto& file : collect_tracked_files(repo)) {
    auto rel = relative_string(repo, file.parent_path());
    while (durable_dir(rel)) {
      dirs.insert(rel);
      rel = std::filesystem::path(rel).parent_path().string();
    }
  }
  for (const auto& rel : dirs) {
    check_readme_child_mentions(repo, repo / rel, dirs, &result);
  }
  return result.errors == 0 ? 0 : 1;
}

int check_no_node(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_tracked_files(repo)) {
    auto name = file.filename().string();
    if (name == "package.json" || name == "package-lock.json" ||
        name == "pnpm-lock.yaml" || name == "yarn.lock" ||
        name == "bun.lockb" || name == "tsconfig.json" ||
        name.rfind("vite.config.", 0) == 0 ||
        name.rfind("webpack.config.", 0) == 0) {
      result.fail("Node runtime artifact is forbidden: " + file.string());
    }
  }
  return result.errors == 0 ? 0 : 1;
}

int check_native_only(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_tracked_files(repo)) {
    check_native_only_file(repo, file, &result);
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
