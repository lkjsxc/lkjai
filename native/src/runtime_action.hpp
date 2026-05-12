#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace lkjai {

struct AgentAction {
  std::string raw;
  std::string tool;
  std::string reasoning;
  std::unordered_map<std::string, std::string> fields;
};

bool parse_agent_action(std::string_view text, AgentAction* action,
                        std::string* error);
std::string agent_action_field(const AgentAction& action,
                               const std::string& name);
std::string agent_action_signature(const AgentAction& action);

}  // namespace lkjai
