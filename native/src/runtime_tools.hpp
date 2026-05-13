#pragma once

#include <string>

#include "runtime_action.hpp"
#include "runtime_api.hpp"

namespace lkjai {

struct RuntimeToolResult {
  bool supported = false;
  std::string tool;
  std::string json;
};

RuntimeToolResult runtime_run_tool(const RuntimeConfig& cfg,
                                   const AgentAction& action);

}  // namespace lkjai
