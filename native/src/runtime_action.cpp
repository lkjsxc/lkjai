#include "runtime_action.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>
#include <vector>

namespace lkjai {
namespace {

std::string trim(std::string_view value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
  size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
  return std::string(value.substr(begin, end - begin));
}

bool name_ok(std::string_view name) {
  if (name.empty()) return false;
  for (unsigned char ch : name) {
    if (!std::isalnum(ch) && ch != '_' && ch != '-') return false;
  }
  return true;
}

bool read_child(std::string_view body, size_t* pos, AgentAction* action,
                std::string* error) {
  while (*pos < body.size() &&
         std::isspace(static_cast<unsigned char>(body[*pos]))) ++*pos;
  if (*pos >= body.size()) return false;
  if (body[*pos] != '<') {
    *error = "action children must be XML tags";
    return false;
  }
  auto open_end = body.find('>', *pos);
  if (open_end == std::string_view::npos) {
    *error = "unterminated action child tag";
    return false;
  }
  auto name = body.substr(*pos + 1, open_end - *pos - 1);
  if (!name_ok(name)) {
    *error = "action child tags must be paired and attribute-free";
    return false;
  }
  auto close = "</" + std::string(name) + ">";
  auto close_pos = body.find(close, open_end + 1);
  if (close_pos == std::string_view::npos) {
    *error = "missing closing tag for " + std::string(name);
    return false;
  }
  auto value = std::string(body.substr(open_end + 1, close_pos - open_end - 1));
  auto key = std::string(name);
  if (action->fields.contains(key)) {
    *error = "duplicate action field " + key;
    return false;
  }
  action->fields[key] = value;
  if (key == "tool") action->tool = trim(value);
  if (key == "reasoning") action->reasoning = trim(value);
  *pos = close_pos + close.size();
  return true;
}

}  // namespace

bool parse_agent_action(std::string_view text, AgentAction* action,
                        std::string* error) {
  AgentAction parsed;
  parsed.raw = trim(text);
  if (!parsed.raw.starts_with("<action>") ||
      !parsed.raw.ends_with("</action>")) {
    *error = "assistant content must be exactly one action block";
    return false;
  }
  auto inner = std::string_view(parsed.raw).substr(
      8, parsed.raw.size() - std::string("</action>").size() - 8);
  size_t pos = 0;
  while (true) {
    while (pos < inner.size() &&
           std::isspace(static_cast<unsigned char>(inner[pos]))) ++pos;
    if (pos >= inner.size()) break;
    if (!read_child(inner, &pos, &parsed, error)) return false;
  }
  if (parsed.tool.empty()) {
    *error = "action missing required tool";
    return false;
  }
  *action = std::move(parsed);
  return true;
}

std::string agent_action_field(const AgentAction& action,
                               const std::string& name) {
  auto found = action.fields.find(name);
  return found == action.fields.end() ? "" : found->second;
}

std::string agent_action_signature(const AgentAction& action) {
  std::vector<std::string> keys;
  for (const auto& [key, _] : action.fields) keys.push_back(key);
  std::sort(keys.begin(), keys.end());
  std::ostringstream out;
  out << action.tool;
  for (const auto& key : keys) {
    if (key == "reasoning") continue;
    out << "\n" << key << "=" << action.fields.at(key);
  }
  return out.str();
}

}  // namespace lkjai
