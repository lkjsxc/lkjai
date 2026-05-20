#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json_min.hpp"
#include "runtime_agent.hpp"
#include "runtime_events.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

lkjai::RuntimeConfig cfg(const std::string& name) {
  auto root = std::filesystem::path("/tmp") / name;
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "workspace");
  lkjai::RuntimeConfig c{"127.0.0.1", 8080, root.string(),
                         "local-native-engine", "agent-model"};
  c.workspace_dir = (root / "workspace").string();
  return c;
}

lkjai::NativeHttpResponse ok(const std::string& content) {
  return {200, "{\"choices\":[{\"message\":{\"content\":\"" +
                   lkjai::json_escape(content) + "\"}}]}",
          ""};
}

lkjai::HttpResponse run(lkjai::RuntimeConfig c,
                        std::vector<lkjai::NativeHttpResponse> responses,
                        const std::string& body) {
  size_t i = 0;
  return lkjai::runtime_chat_with_model_callback(
      c, {"POST", "/api/chat", body}, [&](const std::string& payload) {
        if (!has(payload, "\"role\":\"system\"") ||
            !has(payload, "\"content\":\"hello\"")) {
          return lkjai::NativeHttpResponse{500, "", "bad payload"};
        }
        if (i >= responses.size()) return lkjai::NativeHttpResponse{500, "", "empty"};
        return responses[i++];
      });
}

bool finish_contract() {
  auto resp = run(cfg("lkjai-agent-finish"),
                  {ok("<action><reasoning>done</reasoning>"
                      "<tool>agent.finish</tool><content>hi</content></action>")},
                  "{\"message\":\"hello\",\"run_id\":\"r1\"}");
  return expect(has(resp.body, "\"stop_reason\":\"finish\""), "finish stop") &&
         expect(has(resp.body, "\"assistant\":\"hi\""), "finish answer") &&
         expect(has(resp.body, "\"kind\":\"reasoning\""), "reasoning event") &&
         expect(has(resp.body, "\"kind\":\"finish\""), "finish event");
}

bool think_then_finish_contract() {
  auto resp = run(cfg("lkjai-agent-think"),
                  {ok("<action><tool>agent.think</tool>"
                      "<content>plan</content></action>"),
                   ok("<action><tool>agent.finish</tool>"
                      "<content>done</content></action>")},
                  "{\"message\":\"hello\",\"run_id\":\"r2\",\"max_steps\":2}");
  return expect(has(resp.body, "\"stop_reason\":\"finish\""), "think finish") &&
         expect(has(resp.body, "\"kind\":\"plan\""), "plan event") &&
         expect(has(resp.body, "\"assistant\":\"done\""), "done answer");
}

bool repeat_contract() {
  auto action = ok("<action><tool>agent.think</tool><content>same</content></action>");
  auto resp = run(cfg("lkjai-agent-repeat"), {action, action},
                  "{\"message\":\"hello\",\"run_id\":\"r3\",\"max_steps\":2}");
  return expect(has(resp.body, "\"stop_reason\":\"repeat_action\""),
                "repeat stop");
}

bool error_contracts() {
  auto invalid = run(cfg("lkjai-agent-invalid"), {ok("plain text")},
                     "{\"message\":\"hello\",\"run_id\":\"r4\"}");
  auto unsupported =
      run(cfg("lkjai-agent-tool"),
          {ok("<action><tool>memory.search</tool><query>x</query></action>")},
          "{\"message\":\"hello\",\"run_id\":\"r5\"}");
  return expect(has(invalid.body, "\"stop_reason\":\"invalid_action\""),
                "invalid action") &&
         expect(has(unsupported.body, "\"stop_reason\":\"tool_error\""),
                "tool error");
}

bool filesystem_tool_contract() {
  auto c = cfg("lkjai-agent-fs");
  std::ofstream(std::filesystem::path(c.workspace_dir) / "note.txt") << "alpha";
  auto resp = run(
      c,
      {ok("<action><tool>fs.list</tool><path>.</path></action>"),
       ok("<action><tool>fs.read</tool><path>note.txt</path></action>"),
       ok("<action><tool>agent.finish</tool><content>done</content></action>")},
      "{\"message\":\"hello\",\"run_id\":\"r6\",\"max_steps\":3}");
  auto escape = run(
      c, {ok("<action><tool>fs.read</tool><path>../secret</path></action>"),
          ok("<action><tool>agent.finish</tool><content>blocked</content></action>")},
      "{\"message\":\"hello\",\"run_id\":\"r7\",\"max_steps\":2}");
  return expect(has(resp.body, "\"stop_reason\":\"finish\""), "fs finish") &&
         expect(has(resp.body, "\"kind\":\"tool_call\""), "tool call event") &&
         expect(has(resp.body, "\"kind\":\"tool_result\""), "tool result event") &&
         expect(has(resp.body, "\"kind\":\"observation\""), "observation event") &&
         expect(has(resp.body, "\\\"entries\\\":[\\\"note.txt\\\"]"),
                "list entries") &&
         expect(has(resp.body, "\\\"content\\\":\\\"alpha\\\""),
                "read content") &&
         expect(has(escape.body, "\\\"status\\\":\\\"error\\\""),
                "escape error result") &&
         expect(has(escape.body, "path escapes workspace"),
                "escape rejected");
}

bool tool_profile_contract() {
  auto c = cfg("lkjai-agent-disabled-fs");
  c.tool_profile = "disabled";
  std::ofstream(std::filesystem::path(c.workspace_dir) / "note.txt") << "alpha";
  auto resp = run(
      c, {ok("<action><tool>fs.read</tool><path>note.txt</path></action>"),
          ok("<action><tool>agent.finish</tool><content>blocked</content></action>")},
      "{\"message\":\"hello\",\"run_id\":\"r8\",\"max_steps\":2}");
  return expect(has(resp.body, "\\\"status\\\":\\\"error\\\""),
                "disabled profile error result") &&
         expect(has(resp.body, "tool profile is disabled"),
                "disabled profile rejected before dispatch");
}

bool confirmation_contract() {
  auto c = cfg("lkjai-agent-confirm");
  auto ask = ok("<action><tool>agent.request_confirmation</tool>"
                "<pending_tool>resource.update_resource</pending_tool>"
                "<ref>note-1</ref><body>updated</body></action>");
  auto requested = run(c, {ask}, "{\"message\":\"hello\",\"run_id\":\"r9\"}");
  auto confirmed = lkjai::runtime_chat_with_model_callback(
      c, {"POST", "/api/chat",
          "{\"message\":\"yes\",\"run_id\":\"r9\",\"confirm_pending\":true}"},
      [](const std::string&) {
        return lkjai::NativeHttpResponse{500, "", "model should not run"};
      });
  auto cancelled = lkjai::runtime_chat_with_model_callback(
      c, {"POST", "/api/chat",
          "{\"message\":\"no\",\"run_id\":\"r9\",\"cancel_pending\":true}"},
      [](const std::string&) {
        return lkjai::NativeHttpResponse{500, "", "model should not run"};
      });
  return expect(has(requested.body, "\"stop_reason\":\"confirmation_required\""),
                "confirmation required") &&
         expect(has(requested.body, "\"kind\":\"pending_operation\""),
                "pending persisted") &&
         expect(has(confirmed.body, "KJXLKJ_BEARER_TOKEN not configured"),
                "missing token degraded") &&
         expect(has(cancelled.body, "\"stop_reason\":\"cancelled\""),
                "cancel stop");
}

}  // namespace

int main() {
  return finish_contract() && think_then_finish_contract() && repeat_contract() &&
                 error_contracts() && filesystem_tool_contract() &&
                 tool_profile_contract() && confirmation_contract()
             ? 0
             : 1;
}
