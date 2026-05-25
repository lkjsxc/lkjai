#include "runtime_agent.hpp"
#include "json_min.hpp"
#include "runtime_action.hpp"
#include "runtime_events.hpp"
#include "runtime_tool_registry.hpp"
#include "runtime_tools.hpp"
#include <fstream>
namespace lkjai {
namespace {
std::string error_json(std::string_view error) {
  return "{\"error\":\"" + json_escape(error) + "\"}";
}
int max_steps(std::string_view body, std::string* error) {
  int value = json_int_value(body, "max_steps", 6);
  if (value >= 1 && value <= 64) return value;
  *error = "max_steps must be in [1,64]";
  return 0;
}
std::string chat_payload(const RuntimeConfig& cfg, const std::string& run_id) {
  return "{\"model\":\"" + json_escape(cfg.model) + "\",\"messages\":["
         "{\"role\":\"system\",\"content\":\"" +
         json_escape(runtime_tool_system_prompt(cfg)) + "\"}" +
         runtime_chat_messages_json(cfg, run_id, 12) +
         "],\"max_tokens\":512,\"temperature\":0.2}";
}
std::string choice_content(std::string_view body) {
  auto choices = body.find("\"choices\"");
  if (choices == std::string_view::npos) return "";
  auto message = body.find("\"message\"", choices);
  if (message == std::string_view::npos) return "";
  return json_first_string(body.substr(message), "content");
}
HttpResponse response(const RuntimeConfig& cfg, const std::string& run_id,
                      const std::vector<std::string>& visible,
                      const std::string& assistant, const std::string& stop_reason) {
  return {200, "{\"run_id\":\"" + json_escape(run_id) + "\",\"assistant\":\"" +
                   json_escape(assistant) + "\",\"events\":" +
                   runtime_events_json(cfg, run_id, visible) +
                   ",\"stop_reason\":\"" + stop_reason + "\"}"};
}
HttpResponse stop_error(const RuntimeConfig& cfg, const std::string& run_id,
                        const std::vector<std::string>& visible,
                        const std::string& reason, const std::string& content) {
  runtime_append_event(cfg, run_id, "error", content);
  return response(cfg, run_id, visible, "", reason);
}
void append_reasoning(const RuntimeConfig& cfg, const std::string& run_id,
                      const AgentAction& action, int step) {
  if (!action.reasoning.empty()) {
    runtime_append_event(cfg, run_id, "reasoning", action.reasoning, step);
  }
}
AgentAction pending_action(const AgentAction& confirmation) {
  AgentAction out = confirmation;
  auto pending = agent_action_field(confirmation, "pending_tool");
  if (pending.empty()) pending = agent_action_field(confirmation, "operation");
  out.tool = pending;
  out.fields["tool"] = pending;
  return out;
}
bool last_pending(const RuntimeConfig& cfg, const std::string& run_id,
                  AgentAction* out, std::string* error) {
  std::ifstream file(runtime_run_path(cfg, run_id));
  std::string line, raw;
  while (std::getline(file, line)) {
    if (json_first_string(line, "kind") == "pending_operation") {
      raw = json_first_string(line, "content");
    }
  }
  if (raw.empty()) {
    *error = "no pending operation";
    return false;
  }
  AgentAction parsed;
  if (!parse_agent_action(raw, &parsed, error)) return false;
  *out = parsed.tool == "agent.request_confirmation" ? pending_action(parsed)
                                                      : parsed;
  return true;
}
HttpResponse handle_confirmation(const RuntimeConfig& cfg,
                                 const std::string& run_id,
                                 const std::vector<std::string>& visible,
                                 bool confirm) {
  if (!confirm) {
    runtime_append_event(cfg, run_id, "cancelled", "pending operation cancelled");
    return response(cfg, run_id, visible, "", "cancelled");
  }
  std::string error;
  AgentAction action;
  if (!last_pending(cfg, run_id, &action, &error)) {
    return stop_error(cfg, run_id, visible, "tool_error", error);
  }
  runtime_append_event(cfg, run_id, "confirmed_operation", action.raw, 0,
                       action.tool);
  action.fields["confirmed"] = "true";
  auto result = runtime_run_tool(cfg, action);
  if (!result.supported) {
    return stop_error(cfg, run_id, visible, "tool_error",
                      "unsupported tool: " + action.tool);
  }
  runtime_append_event(cfg, run_id, "tool_result", result.json, 0, action.tool);
  runtime_append_event(cfg, run_id, "observation", result.json, 0, action.tool);
  return response(cfg, run_id, visible, result.json, "finish");
}
}  // namespace
HttpResponse runtime_chat_with_model_callback(const RuntimeConfig& cfg,
                                              const HttpRequest& request,
                                              RuntimeModelCall model_call) {
  auto message = json_first_string(request.body, "message");
  if (message.empty()) return {400, error_json("message is required")};
  std::string error;
  int steps = max_steps(request.body, &error);
  if (steps == 0) return {400, error_json(error)};
  auto run_id = json_first_string(request.body, "run_id");
  if (run_id.empty()) run_id = runtime_new_run_id();
  if (!runtime_run_id_ok(run_id)) return {400, error_json("invalid run_id")};
  runtime_append_event(cfg, run_id, "user", message);
  auto visible = runtime_visible_event_kinds(request.body);
  if (json_bool_value(request.body, "confirm_pending", false)) {
    return handle_confirmation(cfg, run_id, visible, true);
  }
  if (json_bool_value(request.body, "cancel_pending", false)) {
    return handle_confirmation(cfg, run_id, visible, false);
  }
  std::string previous_nonterminal;
  for (int step = 1; step <= steps; ++step) {
    auto model = model_call(chat_payload(cfg, run_id));
    if (model.status != 200) {
      auto body = model.body.empty() ? model.error : model.body;
      return stop_error(cfg, run_id, visible, "model_error", body);
    }
    auto content = choice_content(model.body);
    if (content.empty()) {
      return stop_error(cfg, run_id, visible, "invalid_model_response",
                        "model response missing assistant content");
    }
    AgentAction action;
    if (!parse_agent_action(content, &action, &error)) {
      return stop_error(cfg, run_id, visible, "invalid_action", error);
    }
    append_reasoning(cfg, run_id, action, step);
    if (!runtime_tool_available(cfg, action.tool)) {
      return stop_error(cfg, run_id, visible, "tool_error",
                        "tool not available in profile: " + action.tool);
    }
    if (action.tool != "agent.finish") {
      auto sig = agent_action_signature(action);
      if (!previous_nonterminal.empty() && previous_nonterminal == sig) {
        return stop_error(cfg, run_id, visible, "repeat_action",
                          "repeated non-terminal action");
      }
      previous_nonterminal = sig;
    }
    if (action.tool == "agent.finish") {
      auto answer = agent_action_field(action, "content");
      runtime_append_event(cfg, run_id, "finish", answer, step, action.tool);
      runtime_append_event(cfg, run_id, "assistant", answer, step);
      return response(cfg, run_id, visible, answer, "finish");
    }
    if (action.tool == "agent.think") {
      runtime_append_event(cfg, run_id, "plan",
                           agent_action_field(action, "content"), step,
                           action.tool);
      continue;
    }
    if (action.tool == "agent.request_confirmation" ||
        runtime_tool_requires_confirmation(action.tool)) {
      runtime_append_event(cfg, run_id, "pending_operation", action.raw, step,
                           action.tool);
      return response(cfg, run_id, visible, "", "confirmation_required");
    }
    runtime_append_event(cfg, run_id, "tool_call", action.raw, step,
                         action.tool);
    auto tool = runtime_run_tool(cfg, action);
    if (!tool.supported) {
      return stop_error(cfg, run_id, visible, "tool_error",
                        "unsupported tool: " + action.tool);
    }
    runtime_append_event(cfg, run_id, "tool_result", tool.json, step,
                         action.tool);
    runtime_append_event(cfg, run_id, "observation", tool.json, step,
                         action.tool);
  }
  return stop_error(cfg, run_id, visible, "max_steps",
                    "agent loop reached max_steps");
}
HttpResponse runtime_chat_with_model_response(const RuntimeConfig& cfg,
                                              const HttpRequest& request,
                                              const NativeHttpResponse& model) {
  bool used = false;
  return runtime_chat_with_model_callback(
      cfg, request, [&](const std::string&) {
        if (used) return NativeHttpResponse{500, "", "no model response"};
        used = true;
        return model;
      });
}
}  // namespace lkjai
