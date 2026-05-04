#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace lkjai {

struct RepoCheckResult {
  int errors = 0;
  void fail(const std::string& message);
};

using PathList = std::vector<std::filesystem::path>;

PathList collect_files(const std::filesystem::path& root);
bool is_ignored_path(const std::filesystem::path& path);
int check_docs_topology(const std::filesystem::path& repo);
int check_docs_links(const std::filesystem::path& repo);
int check_line_limits(const std::filesystem::path& repo);
int check_no_node(const std::filesystem::path& repo);
int check_corpus_actions(const std::vector<std::filesystem::path>& paths);

}  // namespace lkjai
