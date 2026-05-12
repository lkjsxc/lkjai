#include <iostream>
#include <string>

#include "runtime_action.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

bool parse_finish() {
  lkjai::AgentAction action;
  std::string error;
  bool ok = lkjai::parse_agent_action(
      "<action>\n<reasoning>Done.</reasoning>\n<tool>agent.finish</tool>\n"
      "<content>Hello.</content>\n</action>",
      &action, &error);
  return expect(ok, error) &&
         expect(action.tool == "agent.finish", "finish tool") &&
         expect(action.reasoning == "Done.", "reasoning field") &&
         expect(lkjai::agent_action_field(action, "content") == "Hello.",
                "content field");
}

bool rejects_bad_shapes() {
  lkjai::AgentAction action;
  std::string error;
  bool missing = !lkjai::parse_agent_action(
      "<action><content>x</content></action>", &action, &error);
  bool attrs = !lkjai::parse_agent_action(
      "<action><tool type=\"x\">agent.finish</tool></action>", &action, &error);
  bool extra = !lkjai::parse_agent_action(
      "x<action><tool>agent.finish</tool></action>", &action, &error);
  bool dup = !lkjai::parse_agent_action(
      "<action><tool>a</tool><tool>b</tool></action>", &action, &error);
  return expect(missing && attrs && extra && dup, "invalid shapes rejected");
}

bool signature_contract() {
  lkjai::AgentAction action;
  std::string error;
  bool ok = lkjai::parse_agent_action(
      "<action><reasoning>a</reasoning><tool>agent.think</tool>"
      "<content>plan</content></action>",
      &action, &error);
  auto sig = lkjai::agent_action_signature(action);
  return expect(ok, error) && expect(has(sig, "agent.think"), "sig tool") &&
         expect(has(sig, "content=plan"), "sig content") &&
         expect(!has(sig, "reasoning="), "sig omits reasoning");
}

}  // namespace

int main() {
  return parse_finish() && rejects_bad_shapes() && signature_contract() ? 0 : 1;
}
