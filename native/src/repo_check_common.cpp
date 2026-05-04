#include "repo_check.hpp"

#include <iostream>

namespace lkjai {

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
        name == "data") {
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

}  // namespace lkjai
