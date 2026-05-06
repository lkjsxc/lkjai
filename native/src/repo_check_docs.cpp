#include "repo_check.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace lkjai {
namespace {

std::string text(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream out;
  out << file.rdbuf();
  return out.str();
}

bool local_link_exists(const std::filesystem::path& base, std::string link) {
  if (link.empty() || link[0] == '#') return true;
  if (link.find("://") != std::string::npos || link.rfind("mailto:", 0) == 0) {
    return true;
  }
  auto hash = link.find('#');
  if (hash != std::string::npos) link.resize(hash);
  auto target = (base / link).lexically_normal();
  return std::filesystem::exists(target);
}

int child_count(const std::filesystem::path& dir) {
  int count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().filename() == "README.md") continue;
    if (entry.is_directory() ||
        (entry.is_regular_file() && entry.path().extension() == ".md")) {
      ++count;
    }
  }
  return count;
}

void check_readme_links(const std::filesystem::path& dir,
                        RepoCheckResult* result) {
  auto readme = text(dir / "README.md");
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    auto path = entry.path();
    if (path.filename() == "README.md") continue;
    if (entry.is_regular_file() && path.extension() != ".md") continue;
    if (!entry.is_directory() && !entry.is_regular_file()) continue;
    auto name = path.filename().string();
    if (readme.find(name) == std::string::npos) {
      result->fail((dir / "README.md").string() + " missing child " + name);
    }
  }
}

}  // namespace

int check_docs_topology(const std::filesystem::path& repo) {
  RepoCheckResult result;
  auto docs = repo / "docs";
  for (std::filesystem::recursive_directory_iterator it(docs), end; it != end;
       ++it) {
    if (!it->is_directory()) continue;
    auto dir = it->path();
    if (!std::filesystem::is_regular_file(dir / "README.md")) {
      result.fail("missing README.md in " + dir.string());
      continue;
    }
    if (child_count(dir) < 2) {
      result.fail("docs directory needs at least two children: " + dir.string());
    }
    check_readme_links(dir, &result);
  }
  return result.errors == 0 ? 0 : 1;
}

int check_docs_links(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& file : collect_files(repo / "docs")) {
    if (file.extension() != ".md") continue;
    auto body = text(file);
    size_t pos = 0;
    while ((pos = body.find("](", pos)) != std::string::npos) {
      auto start = pos + 2;
      auto end = body.find(')', start);
      if (end == std::string::npos) break;
      auto link = body.substr(start, end - start);
      if (!local_link_exists(file.parent_path(), link)) {
        result.fail(file.string() + " broken link " + link);
      }
      pos = end + 1;
    }
  }
  return result.errors == 0 ? 0 : 1;
}

int check_docs_contract_owners(const std::filesystem::path& repo) {
  RepoCheckResult result;
  auto inventory = repo / "docs/architecture/native/contract-inventory.md";
  auto body = text(inventory);
  for (auto required : {"contract_id", "owner", "state", "canonical_source",
                        "supersedes"}) {
    if (body.find(required) == std::string::npos) {
      result.fail(inventory.string() + " missing " + required);
    }
  }

  for (auto relative : {
           "docs/architecture/native/contract-inventory.md",
           "docs/architecture/native/decoder/config.md",
	           "docs/architecture/native/decoder/training.md",
	           "docs/architecture/native/decoder/decode.md",
	           "docs/architecture/native/runtime.md",
	           "docs/architecture/training/packed-cache.md",
	           "docs/architecture/training/dataset.md",
	           "docs/architecture/training/provenance.md",
	           "docs/architecture/agent/schema.md",
	           "docs/architecture/agent/loop.md",
	           "docs/architecture/model/serving.md",
	           "docs/product/api.md",
	           "docs/product/agent-tools.md",
	           "docs/product/chat.md",
	           "docs/operations/compose.md",
	           "docs/operations/training/long-run.md",
	           "docs/operations/performance/benchmarking.md",
	       }) {
    auto path = repo / relative;
    auto contract = text(path);
    if (contract.find("Owner:") == std::string::npos) {
      result.fail(path.string() + " missing Owner marker");
    }
    if (contract.find("State:") == std::string::npos) {
      result.fail(path.string() + " missing State marker");
    }
  }
  return result.errors == 0 ? 0 : 1;
}

}  // namespace lkjai
