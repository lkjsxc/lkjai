#include "runtime_tools.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "json_min.hpp"

namespace lkjai {
namespace {

constexpr size_t kReadLimit = 8192;
constexpr size_t kEntryLimit = 200;

std::string result_json(const std::string& tool, const std::string& status,
                        const std::string& path, const std::string& key,
                        const std::string& value, bool truncated) {
  return "{\"tool\":\"" + json_escape(tool) + "\",\"status\":\"" +
         json_escape(status) + "\",\"path\":\"" + json_escape(path) +
         "\"," + key + ":" + value + ",\"truncated\":" +
         (truncated ? "true" : "false") + "}";
}

std::string error_json(const std::string& tool, const std::string& path,
                       const std::string& error) {
  return result_json(tool, "error", path, "\"error\"",
                     "\"" + json_escape(error) + "\"", false);
}

bool inside(const std::filesystem::path& base,
            const std::filesystem::path& path) {
  auto b = base.lexically_normal();
  auto p = path.lexically_normal();
  auto bit = b.begin();
  auto pit = p.begin();
  for (; bit != b.end(); ++bit, ++pit) {
    if (pit == p.end() || *bit != *pit) return false;
  }
  return true;
}

bool resolve_workspace_path(const RuntimeConfig& cfg, const std::string& raw,
                            std::filesystem::path* out, std::string* error) {
  auto rel = std::filesystem::path(raw.empty() ? "." : raw);
  if (rel.is_absolute()) {
    *error = "absolute paths are not allowed";
    return false;
  }
  std::error_code ec;
  auto root = cfg.workspace_dir.empty() ? std::filesystem::current_path()
                                        : std::filesystem::path(cfg.workspace_dir);
  auto base = std::filesystem::weakly_canonical(root, ec);
  if (ec) {
    *error = "workspace is not available";
    return false;
  }
  auto joined = std::filesystem::weakly_canonical(base / rel, ec);
  if (ec) joined = (base / rel).lexically_normal();
  if (!inside(base, joined)) {
    *error = "path escapes workspace";
    return false;
  }
  *out = joined;
  return true;
}

std::string display_path(const std::filesystem::path& base,
                         const std::filesystem::path& path) {
  auto rel = path.lexically_relative(base);
  auto text = rel.empty() ? "." : rel.string();
  return text.empty() ? "." : text;
}

RuntimeToolResult fs_read(const RuntimeConfig& cfg, const AgentAction& action) {
  auto raw = agent_action_field(action, "path");
  RuntimeToolResult out{true, action.tool, ""};
  if (raw.empty()) {
    out.json = error_json(action.tool, "", "path is required");
    return out;
  }
  std::filesystem::path target;
  std::string error;
  if (!resolve_workspace_path(cfg, raw, &target, &error)) {
    out.json = error_json(action.tool, raw, error);
    return out;
  }
  std::error_code ec;
  auto root = cfg.workspace_dir.empty() ? std::filesystem::current_path()
                                        : std::filesystem::path(cfg.workspace_dir);
  auto base = std::filesystem::weakly_canonical(root, ec);
  auto shown = display_path(base, target);
  if (!std::filesystem::is_regular_file(target, ec)) {
    out.json = error_json(action.tool, shown, "file not found");
    return out;
  }
  if (ec) {
    out.json = error_json(action.tool, shown, "file not found");
    return out;
  }
  std::ifstream file(target, std::ios::binary);
  if (!file) {
    out.json = error_json(action.tool, shown, std::strerror(errno));
    return out;
  }
  std::string content(kReadLimit + 1, '\0');
  file.read(content.data(), static_cast<std::streamsize>(content.size()));
  auto count = static_cast<size_t>(file.gcount());
  bool truncated = count > kReadLimit;
  content.resize(std::min(count, kReadLimit));
  out.json = result_json(action.tool, "ok", shown, "\"content\"",
                         "\"" + json_escape(content) + "\"", truncated);
  return out;
}

RuntimeToolResult fs_list(const RuntimeConfig& cfg, const AgentAction& action) {
  auto raw = agent_action_field(action, "path");
  RuntimeToolResult out{true, action.tool, ""};
  std::filesystem::path target;
  std::string error;
  if (!resolve_workspace_path(cfg, raw.empty() ? "." : raw, &target, &error)) {
    out.json = error_json(action.tool, raw, error);
    return out;
  }
  std::error_code ec;
  auto root = cfg.workspace_dir.empty() ? std::filesystem::current_path()
                                        : std::filesystem::path(cfg.workspace_dir);
  auto base = std::filesystem::weakly_canonical(root, ec);
  auto shown = display_path(base, target);
  if (!std::filesystem::is_directory(target, ec)) {
    out.json = error_json(action.tool, shown, "directory not found");
    return out;
  }
  if (ec) {
    out.json = error_json(action.tool, shown, "directory not found");
    return out;
  }
  std::vector<std::string> entries;
  std::filesystem::directory_iterator it(target, ec);
  if (ec) {
    out.json = error_json(action.tool, shown, ec.message());
    return out;
  }
  for (const auto& entry : it) {
    auto name = entry.path().filename().string();
    std::error_code type_ec;
    if (entry.is_directory(type_ec)) name += "/";
    entries.push_back(name);
  }
  std::sort(entries.begin(), entries.end());
  bool truncated = entries.size() > kEntryLimit;
  std::ostringstream list;
  list << "[";
  for (size_t i = 0; i < entries.size() && i < kEntryLimit; ++i) {
    if (i) list << ",";
    list << "\"" << json_escape(entries[i]) << "\"";
  }
  list << "]";
  out.json = result_json(action.tool, "ok", shown, "\"entries\"", list.str(),
                         truncated);
  return out;
}

}  // namespace

RuntimeToolResult runtime_run_tool(const RuntimeConfig& cfg,
                                   const AgentAction& action) {
  if (cfg.tool_profile != "readonly" && cfg.tool_profile != "default") {
    return {true, action.tool,
            error_json(action.tool, "", "tool profile is disabled")};
  }
  if (action.tool == "fs.read") return fs_read(cfg, action);
  if (action.tool == "fs.list") return fs_list(cfg, action);
  return {false, action.tool, ""};
}

}  // namespace lkjai
