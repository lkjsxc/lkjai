#include "repo_check.hpp"

#include <algorithm>
#include <cctype>
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

std::vector<std::string> stale_decoder_terms() {
  return {"host recompute decode", "host prompt recompute"};
}

bool historical_evidence_doc(const std::filesystem::path& repo,
                             const std::filesystem::path& path) {
  auto relative = path.lexically_relative(repo).string();
  return relative.find("/evidence/") != std::string::npos;
}

bool allowed_route_context(const std::string& body, size_t pos) {
  auto start = pos > 8 ? pos - 8 : 0;
  auto end = std::min(body.size(), pos + 24);
  auto around = body.substr(start, end - start);
  return around.find("/v1") != std::string::npos ||
         around.find("_version") != std::string::npos ||
         around.find("api.moonshot.ai/v1") != std::string::npos;
}

bool bounded_word(const std::string& body, size_t pos,
                  const std::string& term) {
  auto word = [](char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
  };
  bool left = pos == 0 || !word(body[pos - 1]);
  auto end = pos + term.size();
  bool right = end >= body.size() || !word(body[end]);
  return left && right;
}

}  // namespace

int check_docs_wording(const std::filesystem::path& repo) {
  RepoCheckResult result;
  auto blocked = blocked_doc_terms();
  for (const auto& file : collect_tracked_files(repo)) {
    if (!docs_markdown(repo, file)) continue;
    auto body = read_text(file);
    auto relative = file.lexically_relative(repo).string();
    if (!historical_evidence_doc(repo, file)) {
      for (const auto& term : stale_decoder_terms()) {
        if (body.find(term) != std::string::npos) {
          result.fail(relative + " contains stale decoder wording: " + term);
        }
      }
    }
    for (const auto& term : blocked) {
      size_t pos = 0;
      while ((pos = body.find(term, pos)) != std::string::npos) {
        if ((term == "version" || term == "Version") &&
            !bounded_word(body, pos, term)) {
          pos += term.size();
          continue;
        }
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
