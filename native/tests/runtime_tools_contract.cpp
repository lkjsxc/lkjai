#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "runtime_tools.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

lkjai::AgentAction action(const std::string& tool, const std::string& path) {
  lkjai::AgentAction out;
  out.tool = tool;
  out.fields["path"] = path;
  return out;
}

bool fs_read_contract() {
  auto root = std::filesystem::temp_directory_path() / "lkjai-runtime-tools";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "docs");
  std::ofstream(root / "docs" / "current-state.md") << "decoder target\n";
  lkjai::RuntimeConfig cfg{"127.0.0.1", 8082, root.string(),
                           "http://inference:8081/v1/chat/completions",
                           "decoder-40m-3070", "readonly", root.string()};
  auto read = lkjai::runtime_run_tool(cfg, action("fs.read",
                                                  "docs/current-state.md"));
  auto escape = lkjai::runtime_run_tool(cfg, action("fs.read", "../secret"));
  auto absolute = lkjai::runtime_run_tool(cfg, action("fs.read", "/etc/passwd"));
  return expect(read.supported, "fs.read supported") &&
         expect(has(read.json, "\"status\":\"ok\""), "read ok") &&
         expect(has(read.json, "decoder target"), "read content") &&
         expect(has(escape.json, "path escapes workspace"), "escape blocked") &&
         expect(has(absolute.json, "absolute paths are not allowed"),
                "absolute blocked");
}

bool memory_search_contract() {
  auto root = std::filesystem::temp_directory_path() / "lkjai-runtime-memory";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "agent" / "memory");
  std::ofstream(root / "agent" / "memory" / "core.jsonl")
      << "{\"content\":\"alpha decoder memory\"}\n"
      << "{\"summary\":\"beta unrelated\"}\n";
  lkjai::RuntimeConfig cfg{"127.0.0.1", 8082, root.string(),
                           "http://inference:8081/v1/chat/completions",
                           "decoder-40m-3070", "readonly",
                           (root / "workspace").string()};
  auto query = action("memory.search", "");
  query.fields["query"] = "decoder";
  auto found = lkjai::runtime_run_tool(cfg, query);
  auto disabled_cfg = cfg;
  disabled_cfg.tool_profile = "disabled";
  auto blocked = lkjai::runtime_run_tool(disabled_cfg, query);
  return expect(found.supported, "memory.search supported") &&
         expect(has(found.json, "\"status\":\"ok\""), "memory search ok") &&
         expect(has(found.json, "alpha decoder memory"), "memory content") &&
         expect(has(blocked.json, "tool not available in profile"),
                "memory blocked when disabled");
}

}  // namespace

int main() { return fs_read_contract() && memory_search_contract() ? 0 : 1; }
