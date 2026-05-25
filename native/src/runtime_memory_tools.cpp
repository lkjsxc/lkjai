#include "runtime_tools.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "json_min.hpp"

namespace lkjai {
namespace {

std::string lower(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

std::string error_json(const AgentAction& action, const std::string& error) {
  return "{\"tool\":\"" + json_escape(action.tool) +
         "\",\"status\":\"error\",\"error\":\"" + json_escape(error) + "\"}";
}

std::string text_for_line(const std::string& line) {
  auto content = json_first_string(line, "content");
  if (!content.empty()) return content;
  content = json_first_string(line, "text");
  if (!content.empty()) return content;
  content = json_first_string(line, "summary");
  return content.empty() ? line : content;
}

}  // namespace

RuntimeToolResult runtime_run_memory_tool(const RuntimeConfig& cfg,
                                          const AgentAction& action) {
  if (action.tool != "memory.search") return {false, action.tool, ""};
  auto query = agent_action_field(action, "query");
  if (query.empty()) query = agent_action_field(action, "q");
  if (query.empty()) return {true, action.tool, error_json(action, "query is required")};
  int limit = 5;
  try {
    auto raw_limit = agent_action_field(action, "limit");
    if (!raw_limit.empty()) limit = std::clamp(std::stoi(raw_limit), 1, 20);
  } catch (...) {
    return {true, action.tool, error_json(action, "limit must be an integer")};
  }
  auto root = std::filesystem::path(cfg.data_dir) / "agent" / "memory";
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  if (std::filesystem::is_directory(root, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
      if (!ec && entry.path().extension() == ".jsonl") files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  auto needle = lower(query);
  std::ostringstream matches;
  matches << "[";
  bool first = true;
  int count = 0;
  bool truncated = false;
  for (const auto& file_path : files) {
    std::ifstream file(file_path);
    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
      ++line_no;
      auto text = text_for_line(line);
      if (lower(text).find(needle) == std::string::npos) continue;
      if (count >= limit) {
        truncated = true;
        break;
      }
      if (!first) matches << ",";
      first = false;
      matches << "{\"source\":\"" << json_escape(file_path.filename().string())
              << "\",\"line\":" << line_no << ",\"content\":\""
              << json_escape(text) << "\"}";
      ++count;
    }
    if (truncated) break;
  }
  matches << "]";
  return {true, action.tool,
          "{\"tool\":\"memory.search\",\"status\":\"ok\",\"query\":\"" +
              json_escape(query) + "\",\"matches\":" + matches.str() +
              ",\"truncated\":" + (truncated ? "true" : "false") + "}"};
}

}  // namespace lkjai
