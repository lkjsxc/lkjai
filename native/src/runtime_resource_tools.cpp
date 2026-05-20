#include "runtime_tools.hpp"

#include <cctype>
#include <sstream>

#include "json_min.hpp"
#include "native_http_client.hpp"

namespace lkjai {
namespace {

std::string encode(std::string_view text) {
  std::ostringstream out;
  const char* hex = "0123456789ABCDEF";
  for (unsigned char ch : text) {
    if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
      out << ch;
    } else {
      out << '%' << hex[ch >> 4] << hex[ch & 15];
    }
  }
  return out.str();
}

std::string base(const RuntimeConfig& cfg) {
  auto url = cfg.kjxlkj_api_url;
  while (!url.empty() && url.back() == '/') url.pop_back();
  return url + "/api/users/" + encode(cfg.kjxlkj_user) + "/resources";
}

RuntimeToolResult wrap(const AgentAction& action, const NativeHttpResponse& r) {
  RuntimeToolResult out{true, action.tool, ""};
  out.json = "{\"tool\":\"" + json_escape(action.tool) + "\",\"status\":" +
             std::to_string(r.status) + ",\"ok\":" +
             (r.status >= 200 && r.status < 300 ? "true" : "false") +
             ",\"body\":" +
             (r.body.empty() ? "\"\"" : "\"" + json_escape(r.body) + "\"") +
             ",\"error\":\"" + json_escape(r.error) + "\"}";
  return out;
}

RuntimeToolResult degraded(const AgentAction& action, std::string_view reason) {
  return {true, action.tool,
          "{\"tool\":\"" + json_escape(action.tool) +
              "\",\"status\":\"error\",\"degraded\":true,\"error\":\"" +
              json_escape(reason) + "\"}"};
}

std::string bool_field(const AgentAction& action, const std::string& key) {
  auto value = agent_action_field(action, key);
  return value == "true" ? "true" : "false";
}

std::string resource_body(const AgentAction& action) {
  return "{\"body\":\"" + json_escape(agent_action_field(action, "body")) +
         "\",\"alias\":\"" + json_escape(agent_action_field(action, "alias")) +
         "\",\"is_favorite\":" + bool_field(action, "is_favorite") +
         ",\"visibility\":\"public\"}";
}

}  // namespace

bool runtime_tool_requires_confirmation(const std::string& tool) {
  return tool == "resource.create" || tool == "resource.create_note" ||
         tool == "resource.create_media" || tool == "resource.update_resource" ||
         tool == "resource.delete";
}

RuntimeToolResult runtime_run_resource_tool(const RuntimeConfig& cfg,
                                            const AgentAction& action) {
  auto tool = action.tool;
  if (tool == "resource.search") {
    auto q = agent_action_field(action, "query");
    if (q.empty()) q = agent_action_field(action, "q");
    auto kind = agent_action_field(action, "kind");
    auto url = base(cfg) + "/search?q=" + encode(q);
    if (!kind.empty()) url += "&kind=" + encode(kind);
    return wrap(action, native_http_json("GET", url, "", cfg.kjxlkj_bearer_token));
  }
  if (tool == "resource.get" || tool == "resource.fetch") {
    auto ref = agent_action_field(action, "ref");
    if (ref.empty()) ref = agent_action_field(action, "id");
    return wrap(action, native_http_json("GET", base(cfg) + "/" + encode(ref),
                                        "", cfg.kjxlkj_bearer_token));
  }
  if (tool == "resource.history") {
    auto ref = agent_action_field(action, "ref");
    if (ref.empty()) ref = agent_action_field(action, "id");
    return wrap(action, native_http_json("GET", base(cfg) + "/" + encode(ref) +
                                        "/history", "",
                                        cfg.kjxlkj_bearer_token));
  }
  if (runtime_tool_requires_confirmation(tool)) {
    if (cfg.kjxlkj_bearer_token.empty()) {
      return degraded(action, "KJXLKJ_BEARER_TOKEN not configured");
    }
    if (tool == "resource.create" || tool == "resource.create_note") {
      return wrap(action, native_http_json("POST", base(cfg) + "/notes",
                                          resource_body(action),
                                          cfg.kjxlkj_bearer_token));
    }
    auto ref = agent_action_field(action, "ref");
    if (ref.empty()) ref = agent_action_field(action, "id");
    if (tool == "resource.delete") {
      return wrap(action, native_http_json("DELETE", base(cfg) + "/" +
                                          encode(ref), "",
                                          cfg.kjxlkj_bearer_token));
    }
    return wrap(action, native_http_json("PUT", base(cfg) + "/" + encode(ref),
                                        resource_body(action),
                                        cfg.kjxlkj_bearer_token));
  }
  return {false, tool, ""};
}

}  // namespace lkjai
