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
PathList collect_tracked_files(const std::filesystem::path& repo);
bool is_ignored_path(const std::filesystem::path& path);
int check_docs_topology(const std::filesystem::path& repo);
int check_docs_links(const std::filesystem::path& repo);
int check_docs_contract_owners(const std::filesystem::path& repo);
int check_line_limits(const std::filesystem::path& repo);
int check_no_node(const std::filesystem::path& repo);
int check_native_only(const std::filesystem::path& repo);
int check_corpus_actions(const std::vector<std::filesystem::path>& paths);
int check_config_contract(const std::filesystem::path& repo);
int check_cuda_arch_contract(const std::filesystem::path& repo);

}  // namespace lkjai
