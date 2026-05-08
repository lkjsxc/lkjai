#include "repo_check.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path repo_root(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--repo") return argv[i + 1];
  }
  return std::filesystem::current_path();
}

std::vector<std::filesystem::path> trailing_paths(int argc, char** argv) {
  std::vector<std::filesystem::path> paths;
  bool after_dash = false;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (after_dash) {
      paths.push_back(arg);
    } else if (arg == "--") {
      after_dash = true;
    }
  }
  return paths;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "--help") {
    std::cerr << "usage: lkjai-native-repo-check COMMAND [--repo DIR]\n"
	              << "commands: docs-topology docs-links docs-contract-owners "
	                 "docs-wording line-limits no-node native-only config-contract "
	                 "stable-identifiers cuda-arch-contract corpus-actions -- FILE...\n";
    return argc < 2 ? 2 : 0;
  }
  std::string command = argv[1];
  auto repo = repo_root(argc, argv);
  if (command == "docs-topology") return lkjai::check_docs_topology(repo);
  if (command == "docs-links") return lkjai::check_docs_links(repo);
  if (command == "docs-contract-owners")
    return lkjai::check_docs_contract_owners(repo);
  if (command == "docs-wording") return lkjai::check_docs_wording(repo);
  if (command == "line-limits") return lkjai::check_line_limits(repo);
  if (command == "no-node") return lkjai::check_no_node(repo);
	  if (command == "native-only") return lkjai::check_native_only(repo);
	  if (command == "stable-identifiers")
	    return lkjai::check_stable_identifiers(repo);
  if (command == "config-contract") return lkjai::check_config_contract(repo);
  if (command == "cuda-arch-contract")
    return lkjai::check_cuda_arch_contract(repo);
  if (command == "corpus-actions") {
    auto paths = trailing_paths(argc, argv);
    if (paths.empty()) {
      std::cerr << "corpus-actions requires files after --\n";
      return 2;
    }
    return lkjai::check_corpus_actions(paths);
  }
  std::cerr << "unknown repo-check command: " << command << "\n";
  return 2;
}
