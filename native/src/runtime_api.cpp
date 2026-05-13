#include "runtime_api.hpp"

#include <filesystem>
#include <sstream>

#include "json_min.hpp"
#include "runtime_agent.hpp"
#include "runtime_events.hpp"

namespace lkjai {
namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + json_escape(error) + "\"}";
}

int query_limit(const std::string& path) {
  auto pos = path.find("limit=");
  if (pos == std::string::npos) return 20;
  try {
    return std::stoi(path.substr(pos + 6));
  } catch (...) {
    return 20;
  }
}

HttpResponse run(const RuntimeConfig& cfg, const std::string& id) {
  if (!runtime_run_id_ok(id)) return {400, error_json("invalid run_id")};
  auto path = runtime_run_path(cfg, id);
  if (!std::filesystem::is_regular_file(path)) return {404, error_json("run not found")};
  return {200, "{\"run_id\":\"" + json_escape(id) + "\",\"events\":" +
                   runtime_events_json(cfg, id, {}) + "}"};
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
  bool loaded = probe.status == 200;
  bool reachable = probe.status > 0;
  std::ostringstream out;
  out << "{\"model\":\"" << json_escape(cfg.model)
      << "\",\"api_url\":\"" << json_escape(cfg.model_url)
      << "\",\"loaded\":" << (loaded ? "true" : "false")
      << ",\"reachable\":" << (reachable ? "true" : "false")
      << ",\"message\":\""
      << (loaded ? "model loaded"
                 : (reachable ? "model server reachable" : "model probe failed"))
      << "\",\"device\":\"" << json_escape(json_first_string(probe.body, "device"))
      << "\",\"cuda_available\":"
      << (json_bool_value(probe.body, "cuda_available", false) ? "true" : "false")
      << ",\"gpu_name\":\"" << json_escape(json_first_string(probe.body, "gpu_name"))
      << "\",\"warning\":\"" << json_escape(json_first_string(probe.body, "warning"))
      << "\",\"probe_status\":" << probe.status
      << ",\"artifact_kind\":\""
      << json_escape(json_first_string(probe.body, "artifact_kind")) << "\""
      << ",\"chat_supported\":"
      << (json_bool_value(probe.body, "chat_supported", false) ? "true" : "false")
      << ",\"decode_supported\":"
      << (json_bool_value(probe.body, "decode_supported", false) ? "true" : "false")
      << ",\"dense_supported\":"
      << (json_first_string(probe.body, "artifact_kind") == "dense" ? "true" : "false")
      << ",\"degraded\":"
      << (json_bool_value(probe.body, "degraded", !loaded) ? "true" : "false")
      << ",\"degraded_reason\":\""
      << json_escape(json_first_string(probe.body, "degraded_reason")) << "\""
      << ",\"tool_profile\":\"" << json_escape(cfg.tool_profile) << "\"}";
  return out.str();
}

std::string runtime_health_json(const RuntimeConfig& cfg) {
  return "{\"status\":\"ok\",\"service\":\"lkjai-native-runtime\",\"bind\":"
         "{\"host\":\"" + json_escape(cfg.host) + "\",\"port\":" +
         std::to_string(cfg.port) + "}}";
}

HttpResponse runtime_route(const RuntimeConfig& cfg, const HttpRequest& request) {
  if (request.method == "OPTIONS") return {204, ""};
  if (request.method == "GET" && request.path == "/healthz") return {200, runtime_health_json(cfg)};
  if (request.method == "GET" && request.path == "/api/model") {
    auto probe = native_http_get(model_url_to_models_url(cfg.model_url));
    if (probe.status != 200) {
      auto health = native_http_get(model_url_to_health_url(cfg.model_url));
      if (health.status == 200) {
        health.status = probe.status;
        probe = health;
      }
    }
    return {200, runtime_model_status_json(cfg, probe)};
  }
  if (request.method == "GET" && request.path == "/api/config") {
    return {200, runtime_config_status_json(cfg)};
  }
  if (request.method == "GET" &&
      (request.path == "/api/runs" || request.path.rfind("/api/runs?", 0) == 0)) {
    return {200, runtime_runs_json(cfg, query_limit(request.path))};
  }
  if (request.method == "POST" && request.path == "/api/chat") {
    return runtime_chat_with_model_callback(
        cfg, request, [&](const std::string& payload) {
          return native_http_post_json(cfg.model_url, payload);
        });
  }
  const std::string prefix = "/api/runs/";
  if (request.method == "GET" && request.path.rfind(prefix, 0) == 0) {
    return run(cfg, request.path.substr(prefix.size()));
  }
  return {404, error_json("not found")};
}

}  // namespace lkjai
