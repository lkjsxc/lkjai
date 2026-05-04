#include "repo_check.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace lkjai {

int check_corpus_actions(const std::vector<std::filesystem::path>& paths) {
  RepoCheckResult result;
  for (const auto& path : paths) {
    std::ifstream file(path);
    if (!file) {
      result.fail("cannot read corpus file " + path.string());
      continue;
    }
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
      ++row;
      if (line.find("<action>") == std::string::npos ||
          line.find("<tool>") == std::string::npos) {
        result.fail(path.string() + ":" + std::to_string(row) +
                    " missing assistant action/tool tags");
      }
      if (line.find("resource.create_") != std::string::npos ||
          line.find("resource.update_resource") != std::string::npos) {
        if (line.find("agent.request_confirmation") == std::string::npos) {
          result.fail(path.string() + ":" + std::to_string(row) +
                      " mutable resource row lacks confirmation");
        }
      }
    }
  }
  if (result.errors == 0) std::cout << "corpus actions ok\n";
  return result.errors == 0 ? 0 : 1;
}

}  // namespace lkjai
