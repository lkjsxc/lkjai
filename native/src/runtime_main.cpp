#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "env.hpp"
#include "http_server.hpp"
#include "json_min.hpp"
#include "native_http_client.hpp"

using lkjai::HttpRequest;
using lkjai::HttpResponse;

namespace {

struct RuntimeConfig {
  std::string host = lkjai::env_string("APP_HOST", "0.0.0.0");
  int port = lkjai::env_int("APP_PORT", 8080);
  std::string data_dir = lkjai::env_string("DATA_DIR", "/app/data");
  std::string model_url = lkjai::env_string(
      "MODEL_API_URL", "http://127.0.0.1:8081/v1/chat/completions");
  std::string model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
};

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + lkjai::json_escape(error) + "\"}";
}

std::string now_id() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  return "run-" + std::to_string(ms);
}

std::filesystem::path run_path(const RuntimeConfig& cfg, const std::string& id) {
  return std::filesystem::path(cfg.data_dir) / "agent" / "runs" / (id + ".jsonl");
}

void append_event(const RuntimeConfig& cfg, const std::string& run_id,
                  const std::string& kind, const std::string& content) {
  auto path = run_path(cfg, run_id);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::app);
  out << "{\"kind\":\"" << lkjai::json_escape(kind) << "\",\"content\":\""
      << lkjai::json_escape(content) << "\"}\n";
}

HttpResponse model_status(const RuntimeConfig& cfg) {
  auto probe = lkjai::native_http_get(lkjai::model_url_to_models_url(cfg.model_url));
  bool ok = probe.status == 200;
  std::ostringstream out;
  out << "{\"model\":\"" << lkjai::json_escape(cfg.model)
      << "\",\"api_url\":\"" << lkjai::json_escape(cfg.model_url)
      << "\",\"loaded\":true,\"reachable\":" << (ok ? "true" : "false")
      << ",\"message\":\"" << (ok ? "model server responding" : "model probe failed")
      << "\",\"probe_status\":" << probe.status << "}";
  return {200, out.str()};
}

std::string chat_payload(const RuntimeConfig& cfg, const std::string& message) {
  return "{\"model\":\"" + lkjai::json_escape(cfg.model) +
         "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
         lkjai::json_escape(message) +
         "\"}],\"max_tokens\":512,\"temperature\":0.2}";
}

HttpResponse chat(const RuntimeConfig& cfg, const HttpRequest& request) {
  auto message = lkjai::json_first_string(request.body, "message");
  if (message.empty()) return {400, error_json("message is required")};
  auto run_id = lkjai::json_first_string(request.body, "run_id");
  if (run_id.empty()) run_id = now_id();
  append_event(cfg, run_id, "user", message);
  auto model = lkjai::native_http_post_json(cfg.model_url, chat_payload(cfg, message));
  if (model.status != 200) {
    append_event(cfg, run_id, "error", model.body.empty() ? model.error : model.body);
    return {200, "{\"run_id\":\"" + lkjai::json_escape(run_id) +
                     "\",\"assistant\":\"\",\"events\":[],"
                     "\"stop_reason\":\"model_error\"}"};
  }
  auto content = lkjai::json_first_string(model.body, "content");
  append_event(cfg, run_id, "assistant", content);
  return {200, "{\"run_id\":\"" + lkjai::json_escape(run_id) +
                   "\",\"assistant\":\"" + lkjai::json_escape(content) +
                   "\",\"events\":[],\"stop_reason\":\"finish\"}"};
}

HttpResponse run(const RuntimeConfig& cfg, const std::string& id) {
  auto path = run_path(cfg, id);
  if (!std::filesystem::is_regular_file(path)) return {404, error_json("run not found")};
  std::ifstream file(path);
  std::ostringstream out;
  out << "{\"run_id\":\"" << lkjai::json_escape(id) << "\",\"events\":[";
  std::string line;
  bool first = true;
  while (std::getline(file, line)) {
    if (!first) out << ",";
    first = false;
    out << line;
  }
  out << "]}";
  return {200, out.str()};
}

HttpResponse route(const RuntimeConfig& cfg, const HttpRequest& request) {
  if (request.method == "GET" && request.path == "/healthz") return {200, "ok"};
  if (request.method == "GET" && request.path == "/") {
    return {200, "{\"service\":\"lkjai-native-runtime\"}"};
  }
  if (request.method == "GET" && request.path == "/api/model") return model_status(cfg);
  if (request.method == "POST" && request.path == "/api/chat") return chat(cfg, request);
  const std::string prefix = "/api/runs/";
  if (request.method == "GET" && request.path.rfind(prefix, 0) == 0) {
    return run(cfg, request.path.substr(prefix.size()));
  }
  return {404, error_json("not found")};
}

}  // namespace

int main() {
  RuntimeConfig cfg;
  return lkjai::serve_http(cfg.host, cfg.port, [&](const HttpRequest& request) {
    return route(cfg, request);
  });
}
