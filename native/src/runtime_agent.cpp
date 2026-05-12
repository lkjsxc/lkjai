#include "runtime_agent.hpp"

#include "json_min.hpp"
#include "runtime_action.hpp"
#include "runtime_events.hpp"

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

std::string system_prompt(const RuntimeConfig& cfg) {
  return "Return exactly one XML action. Available tools: agent.finish, "
         "agent.think. Use <tool>agent.finish</tool> for ordinary replies. "
         "Use <tool>agent.think</tool> only for a short visible plan. "
         "Tool profile: " +
         cfg.tool_profile + ".";
}

std::string chat_payload(const RuntimeConfig& cfg, const std::string& run_id) {
  return "{\"model\":\"" + json_escape(cfg.model) + "\",\"messages\":["
         "{\"role\":\"system\",\"content\":\"" +
         json_escape(system_prompt(cfg)) + "\"}" +
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
                      const std::string& assistant,
                      const std::string& stop_reason) {
  return {200, "{\"run_id\":\"" + json_escape(run_id) + "\",\"assistant\":\"" +
                   json_escape(assistant) + "\",\"events\":" +
                   runtime_events_json(cfg, run_id, visible) +
                   ",\"stop_reason\":\"" + stop_reason + "\"}"};
}

HttpResponse stop_error(const RuntimeConfig& cfg, const std::string& run_id,
                        const std::vector<std::string>& visible,
                        const std::string& reason,
                        const std::string& content) {
  runtime_append_event(cfg, run_id, "error", content);
  return response(cfg, run_id, visible, "", reason);
}

void append_reasoning(const RuntimeConfig& cfg, const std::string& run_id,
                      const AgentAction& action, int step) {
  if (!action.reasoning.empty()) {
    runtime_append_event(cfg, run_id, "reasoning", action.reasoning, step);
  }
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
    runtime_append_event(cfg, run_id, "tool_call", action.raw, step,
                         action.tool);
    return stop_error(cfg, run_id, visible, "tool_error",
                      "unsupported tool: " + action.tool);
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
