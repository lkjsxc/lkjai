#include "runtime_tool_registry.hpp"

#include <iterator>
#include <sstream>

#include "json_min.hpp"
#include "runtime_tools.hpp"

namespace lkjai {
namespace {

const char* kReadonly[] = {"agent.finish", "agent.think", "fs.read", "fs.list",
                           "memory.search", "resource.search", "resource.get",
                           "resource.history"};
const char* kMutable[] = {"agent.request_confirmation", "resource.create",
                          "resource.update_resource", "resource.delete"};
const char* kDisabled[] = {"memory.write", "shell.exec", "web.fetch",
                           "fs.write"};

bool has(const char* const* items, size_t count, const std::string& tool) {
  for (size_t i = 0; i < count; ++i) {
    if (tool == items[i]) return true;
  }
  return false;
}

std::string json_array(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ",";
    out << "\"" << json_escape(values[i]) << "\"";
  }
  out << "]";
  return out.str();
}

std::vector<std::string> disabled_tools() {
  std::vector<std::string> out;
  for (auto* tool : kDisabled) out.push_back(tool);
  return out;
}

}  // namespace

bool runtime_tool_profile_known(const std::string& profile) {
  return profile == "readonly" || profile == "mutable" || profile == "disabled";
}

std::vector<std::string> runtime_available_tools(const RuntimeConfig& cfg) {
  std::vector<std::string> tools;
  tools.push_back("agent.finish");
  if (cfg.tool_profile == "disabled") return tools;
  if (cfg.tool_profile != "readonly" && cfg.tool_profile != "mutable") return tools;
  tools.clear();
  for (auto* tool : kReadonly) tools.push_back(tool);
  if (cfg.tool_profile == "mutable") {
    for (auto* tool : kMutable) tools.push_back(tool);
  }
  return tools;
}

bool runtime_tool_known(const std::string& tool) {
  return has(kReadonly, std::size(kReadonly), tool) ||
         has(kMutable, std::size(kMutable), tool) ||
         has(kDisabled, std::size(kDisabled), tool);
}

bool runtime_tool_available(const RuntimeConfig& cfg, const std::string& tool) {
  auto tools = runtime_available_tools(cfg);
  for (const auto& available : tools) {
    if (available == tool) return true;
  }
  return false;
}

bool runtime_mutable_tools_enabled(const RuntimeConfig& cfg) {
  return cfg.tool_profile == "mutable" && !cfg.kjxlkj_bearer_token.empty();
}

std::string runtime_tool_system_prompt(const RuntimeConfig& cfg) {
  auto tools = runtime_available_tools(cfg);
  std::ostringstream out;
  out << "Return exactly one XML action. Available tools: ";
  for (size_t i = 0; i < tools.size(); ++i) {
    if (i) out << ", ";
    out << tools[i];
  }
  out << ". Tool profile: " << cfg.tool_profile << ". ";
  if (cfg.tool_profile == "mutable") {
    out << "Use agent.request_confirmation before resource mutations. ";
  }
  out << "Use fs.list/fs.read for read-only files. "
      << "Use memory.search only for durable memory lookup.";
  return out.str();
}

std::string runtime_tool_config_json(const RuntimeConfig& cfg) {
  return "{\"profile\":\"" + json_escape(cfg.tool_profile) +
         "\",\"profile_known\":" +
         (runtime_tool_profile_known(cfg.tool_profile) ? "true" : "false") +
         ",\"available\":" + json_array(runtime_available_tools(cfg)) +
         ",\"disabled\":" + json_array(disabled_tools()) +
         ",\"mutable_tools_enabled\":" +
         (runtime_mutable_tools_enabled(cfg) ? "true" : "false") + "}";
}

}  // namespace lkjai
