#include "runtime_api.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "json_min.hpp"
#include "native_status_page.hpp"

namespace lkjai {
namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + json_escape(error) + "\"}";
}

std::string now_id() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  return "run-" + std::to_string(ms);
}

std::string timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::filesystem::path run_path(const RuntimeConfig& cfg, const std::string& id) {
  return std::filesystem::path(cfg.data_dir) / "agent" / "runs" / (id + ".jsonl");
}

void append_event(const RuntimeConfig& cfg, const std::string& run_id,
                  const std::string& kind, const std::string& content,
                  int step = 0, const std::string& tool = "") {
  auto path = run_path(cfg, run_id);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::app);
  out << "{\"kind\":\"" << json_escape(kind) << "\",\"content\":\""
      << json_escape(content) << "\",\"timestamp\":\"" << timestamp() << "\"";
  if (step > 0) out << ",\"step\":" << step;
  if (!tool.empty()) out << ",\"tool\":\"" << json_escape(tool) << "\"";
  out << "}\n";
}

bool includes(const std::vector<std::string>& values, const std::string& value) {
  for (const auto& item : values) if (item == value) return true;
  return false;
}

std::string events_json(const RuntimeConfig& cfg, const std::string& run_id,
                        const std::vector<std::string>& visible) {
  std::ifstream file(run_path(cfg, run_id));
  std::ostringstream out;
  out << "[";
  std::string line;
  bool first = true;
  while (std::getline(file, line)) {
    auto kind = json_first_string(line, "kind");
    if (!visible.empty() && !includes(visible, kind)) continue;
    if (!first) out << ",";
    first = false;
    out << line;
  }
  out << "]";
  return out.str();
}

bool max_steps_ok(std::string_view body, std::string* error) {
  int max_steps = json_int_value(body, "max_steps", 6);
  if (max_steps >= 1 && max_steps <= 64) return true;
  *error = "max_steps must be in [1,64]";
  return false;
}

std::string chat_payload(const RuntimeConfig& cfg, const std::string& message) {
  return "{\"model\":\"" + json_escape(cfg.model) +
         "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
         json_escape(message) + "\"}],\"max_tokens\":512,\"temperature\":0.2}";
}

HttpResponse run(const RuntimeConfig& cfg, const std::string& id) {
  auto path = run_path(cfg, id);
  if (!std::filesystem::is_regular_file(path)) return {404, error_json("run not found")};
  return {200, "{\"run_id\":\"" + json_escape(id) + "\",\"events\":" +
                   events_json(cfg, id, {}) + "}"};
}

}  // namespace

std::vector<std::string> runtime_visible_event_kinds(std::string_view body) {
  std::vector<std::string> kinds;
  auto key = body.find("\"visible_event_kinds\"");
  if (key == std::string_view::npos) return kinds;
  auto open = body.find('[', key);
  auto close = body.find(']', open);
  if (open == std::string_view::npos || close == std::string_view::npos) return kinds;
  auto list = body.substr(open + 1, close - open - 1);
  size_t pos = 0;
  while ((pos = list.find('"', pos)) != std::string_view::npos) {
    auto end = list.find('"', pos + 1);
    if (end == std::string_view::npos) break;
    kinds.emplace_back(list.substr(pos + 1, end - pos - 1));
    pos = end + 1;
  }
  return kinds;
}

std::string runtime_model_status_json(const RuntimeConfig& cfg,
                                      const NativeHttpResponse& probe) {
  bool ok = probe.status == 200;
  std::ostringstream out;
  out << "{\"model\":\"" << json_escape(cfg.model)
      << "\",\"api_url\":\"" << json_escape(cfg.model_url)
      << "\",\"loaded\":" << (!cfg.model_url.empty() ? "true" : "false")
      << ",\"reachable\":" << (ok ? "true" : "false")
      << ",\"message\":\"" << (ok ? "model server responding" : "model probe failed")
      << "\",\"device\":\"" << json_escape(json_first_string(probe.body, "device"))
      << "\",\"cuda_available\":"
      << (json_bool_value(probe.body, "cuda_available", false) ? "true" : "false")
      << ",\"gpu_name\":\"" << json_escape(json_first_string(probe.body, "gpu_name"))
      << "\",\"warning\":\"" << json_escape(json_first_string(probe.body, "warning"))
      << "\",\"probe_status\":" << probe.status << "}";
  return out.str();
}

HttpResponse runtime_chat_with_model_response(const RuntimeConfig& cfg,
                                              const HttpRequest& request,
                                              const NativeHttpResponse& model) {
  auto message = json_first_string(request.body, "message");
  if (message.empty()) return {400, error_json("message is required")};
  std::string error;
  if (!max_steps_ok(request.body, &error)) return {400, error_json(error)};
  auto run_id = json_first_string(request.body, "run_id");
  if (run_id.empty()) run_id = now_id();
  append_event(cfg, run_id, "user", message);
  auto visible = runtime_visible_event_kinds(request.body);
  if (model.status != 200) {
    append_event(cfg, run_id, "error", model.body.empty() ? model.error : model.body);
    return {200, "{\"run_id\":\"" + json_escape(run_id) +
                     "\",\"assistant\":\"\",\"events\":" +
                     events_json(cfg, run_id, visible) +
                     ",\"stop_reason\":\"model_error\"}"};
  }
  auto content = json_first_string(model.body, "content");
  if (content.empty()) {
    append_event(cfg, run_id, "error", "model response missing assistant content");
    return {200, "{\"run_id\":\"" + json_escape(run_id) +
                     "\",\"assistant\":\"\",\"events\":" +
                     events_json(cfg, run_id, visible) +
                     ",\"stop_reason\":\"invalid_model_response\"}"};
  }
  append_event(cfg, run_id, "assistant", content);
  return {200, "{\"run_id\":\"" + json_escape(run_id) +
                   "\",\"assistant\":\"" + json_escape(content) +
                   "\",\"events\":" + events_json(cfg, run_id, visible) +
                   ",\"stop_reason\":\"finish\"}"};
}

std::string runtime_health_json(const RuntimeConfig& cfg) {
  return "{\"status\":\"ok\",\"service\":\"lkjai-native-runtime\",\"bind\":"
         "{\"host\":\"" + json_escape(cfg.host) + "\",\"port\":" +
         std::to_string(cfg.port) + "}}";
}

HttpResponse runtime_route(const RuntimeConfig& cfg, const HttpRequest& request) {
  if (request.method == "GET" && request.path == "/healthz") return {200, runtime_health_json(cfg)};
  if (request.method == "GET" && request.path == "/") {
    return {200, std::string(native_status_page_html()), "text/html; charset=utf-8"};
  }
  if (request.method == "GET" && request.path == "/api/model") {
    auto probe = native_http_get(model_url_to_models_url(cfg.model_url));
    return {200, runtime_model_status_json(cfg, probe)};
  }
  if (request.method == "GET" && request.path == "/api/config") {
    return {200, runtime_config_status_json(cfg)};
  }
  if (request.method == "POST" && request.path == "/api/chat") {
    auto message = json_first_string(request.body, "message");
    if (message.empty()) return {400, error_json("message is required")};
    std::string error;
    if (!max_steps_ok(request.body, &error)) return {400, error_json(error)};
    auto model = native_http_post_json(cfg.model_url, chat_payload(cfg, message));
    return runtime_chat_with_model_response(cfg, request, model);
  }
  const std::string prefix = "/api/runs/";
  if (request.method == "GET" && request.path.rfind(prefix, 0) == 0) {
    return run(cfg, request.path.substr(prefix.size()));
  }
  return {404, error_json("not found")};
}

}  // namespace lkjai
