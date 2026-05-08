#include "repo_check.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace lkjai {
namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

bool docs_markdown(const std::filesystem::path& repo,
                   const std::filesystem::path& path) {
  return path.extension() == ".md" &&
         path.lexically_relative(repo).string().rfind("docs/", 0) == 0;
}

std::vector<std::string> blocked_doc_terms() {
  return {"roadmap", "Roadmap", "milestone", "Milestone", " P0 ",
          "P0 ",     " P0",      "compatibility", "schema v",
          " v0.",    " v1",      " v2",           "Version",
          "version"};
}

bool allowed_route_context(const std::string& body, size_t pos) {
  auto start = pos > 8 ? pos - 8 : 0;
  auto end = std::min(body.size(), pos + 24);
  auto around = body.substr(start, end - start);
  return around.find("/v1") != std::string::npos ||
         around.find("_version") != std::string::npos ||
         around.find("api.moonshot.ai/v1") != std::string::npos;
}

}  // namespace

int check_docs_wording(const std::filesystem::path& repo) {
  RepoCheckResult result;
  auto blocked = blocked_doc_terms();
  for (const auto& file : collect_tracked_files(repo)) {
    if (!docs_markdown(repo, file)) continue;
    auto body = read_text(file);
    auto relative = file.lexically_relative(repo).string();
    for (const auto& term : blocked) {
      size_t pos = 0;
      while ((pos = body.find(term, pos)) != std::string::npos) {
        if (!allowed_route_context(body, pos)) {
          result.fail(relative + " contains discouraged wording: " + term);
          break;
        }
        pos += term.size();
      }
    }
  }
  return result.errors == 0 ? 0 : 1;
}

}  // namespace lkjai
