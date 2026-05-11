#include "repo_check.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace lkjai {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

bool has(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

void reject_banned_paths(const std::filesystem::path& repo,
                         RepoCheckResult* result) {
  const std::vector<std::string> banned = {
      std::string("/home/lkjsxc/workspace/") + "private/archived/security/" +
          "secrets",
      std::string("private/archived/security/") + "secrets",
      std::string("archived/security/") + "secrets/passwords",
      std::string("legacy-") + "passwords.md"};
  for (const auto& path : collect_tracked_files(repo)) {
    auto body = read_file(path);
    for (auto needle : banned) {
      if (has(body, needle)) {
        result->fail(path.string() + " references unsafe secret path " +
                     std::string(needle));
      }
    }
  }
}

void require_local_hf_default(const std::filesystem::path& repo,
                              RepoCheckResult* result) {
  auto env = read_file(repo / ".env.example");
  auto compose = read_file(repo / "compose.yaml");
  if (!has(env, "HF_SECRET_FILE=./data/secrets/hf_token")) {
    result->fail(".env.example must default HF_SECRET_FILE to data/secrets");
  }
  if (!has(compose, "${HF_SECRET_FILE:-./data/secrets/hf_token}")) {
    result->fail("compose.yaml must default HF_SECRET_FILE to data/secrets");
  }
}

bool checked_secret_surface(const std::filesystem::path& path) {
  auto ext = path.extension().string();
  return ext == ".md" || ext == ".json" || ext == ".yaml" || ext == ".yml" ||
         path.filename() == "manifest.json" || path.filename() == ".env.example";
}

void reject_secret_values_and_stale_words(const std::filesystem::path& repo,
                                          RepoCheckResult* result) {
  const std::regex hf_token(R"(hf_[A-Za-z0-9]{20,})");
  const std::regex bearer(R"(Bearer\s+[A-Za-z0-9._~+\/=-]{16,})");
  const std::regex moonshot(R"(sk-[A-Za-z0-9]{20,})");
  for (const auto& path : collect_tracked_files(repo)) {
    if (!checked_secret_surface(path)) continue;
    auto body = read_file(path);
    if (path.extension() == ".md" && has(body, "secret markdown")) {
      result->fail(path.string() + " uses stale secret markdown wording");
    }
    if (std::regex_search(body, hf_token) || std::regex_search(body, bearer) ||
        std::regex_search(body, moonshot)) {
      result->fail(path.string() + " appears to contain a secret value");
    }
  }
}

}  // namespace

int check_secret_defaults(const std::filesystem::path& repo) {
  RepoCheckResult result;
  reject_banned_paths(repo, &result);
  require_local_hf_default(repo, &result);
  reject_secret_values_and_stale_words(repo, &result);
  if (result.errors == 0) {
    std::cout << "secret defaults ok\n";
  }
  return result.errors == 0 ? 0 : 1;
}

}  // namespace lkjai
