#include "repo_check.hpp"

#include <cstdio>
#include <iostream>

namespace lkjai {
namespace {

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') out += "'\\''";
    else out += ch;
  }
  out += "'";
  return out;
}

}  // namespace

void RepoCheckResult::fail(const std::string& message) {
  ++errors;
  std::cerr << "repo-check: " << message << "\n";
}

bool is_ignored_path(const std::filesystem::path& path) {
  for (const auto& part : path) {
    auto name = part.string();
    if (name == ".git" || name == "target" || name == "build" ||
        name == ".pytest_cache" || name == "__pycache__" ||
        name == "artifacts" || name == "runs" || name == "reports" ||
        name == "data" || name == "tmp") {
      return true;
    }
  }
  return false;
}

PathList collect_files(const std::filesystem::path& root) {
  PathList files;
  for (std::filesystem::recursive_directory_iterator it(root), end; it != end;
       ++it) {
    if (is_ignored_path(it->path())) {
      if (it->is_directory()) it.disable_recursion_pending();
      continue;
    }
    if (it->is_regular_file()) files.push_back(it->path());
  }
  return files;
}

PathList collect_tracked_files(const std::filesystem::path& repo) {
  PathList files;
  auto root = std::filesystem::weakly_canonical(repo);
  auto safe_repo = shell_quote(root.string());
  auto command = "git -c safe.directory=" + safe_repo + " -C " + safe_repo +
                 " ls-files";
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return collect_files(repo);
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe)) {
    std::string line(buffer);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    if (!line.empty()) files.push_back(root / line);
  }
  int status = pclose(pipe);
  return status == 0 ? files : collect_files(root);
}

}  // namespace lkjai
