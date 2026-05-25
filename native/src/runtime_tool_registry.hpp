#pragma once

#include <string>
#include <vector>

#include "runtime_api.hpp"

namespace lkjai {

std::vector<std::string> runtime_available_tools(const RuntimeConfig& cfg);
bool runtime_tool_available(const RuntimeConfig& cfg, const std::string& tool);
bool runtime_tool_known(const std::string& tool);
bool runtime_tool_profile_known(const std::string& profile);
bool runtime_mutable_tools_enabled(const RuntimeConfig& cfg);
std::string runtime_tool_system_prompt(const RuntimeConfig& cfg);
std::string runtime_tool_config_json(const RuntimeConfig& cfg);

}  // namespace lkjai
