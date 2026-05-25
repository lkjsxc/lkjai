#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "runtime_api.hpp"
#include "runtime_events.hpp"

namespace {

bool has(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

int count_runs(const std::string& text) {
  int count = 0;
  size_t pos = 0;
  while ((pos = text.find("\"run_id\"", pos)) != std::string::npos) {
    ++count;
    pos += 8;
  }
  return count;
}

lkjai::RuntimeConfig cfg() {
  auto root = std::filesystem::path("/tmp/lkjai-runtime-contract");
  std::filesystem::remove_all(root);
  return {"127.0.0.1", 8080, root.string(),
          "http://127.0.0.1:8081/v1/chat/completions", "test-model"};
}

bool chat_filter_contract() {
  auto c = cfg();
  lkjai::HttpRequest req;
  req.method = "POST";
  req.path = "/api/chat";
  req.body = "{\"message\":\"hello\",\"run_id\":\"r1\",\"max_steps\":6,"
             "\"visible_event_kinds\":[\"user\"]}";
  lkjai::NativeHttpResponse model;
  model.status = 200;
  model.body =
      "{\"choices\":[{\"message\":{\"content\":\"<action>\\n<tool>agent.finish</tool>\\n<content>hi</content>\\n</action>\"}}]}";
  auto resp = lkjai::runtime_chat_with_model_response(c, req, model);
  auto transcript = std::filesystem::path(c.data_dir) / "agent" / "runs" /
                    "r1.jsonl";
  std::ifstream file(transcript);
  std::string body((std::istreambuf_iterator<char>(file)), {});
  return expect(resp.status == 200, "chat response status") &&
         expect(has(resp.body, "\"stop_reason\":\"finish\""), "finish stop") &&
         expect(has(resp.body, "\"assistant\":\"hi\""), "assistant content") &&
         expect(!has(resp.body, "\"kind\":\"assistant\""), "filtered assistant") &&
         expect(has(body, "\"kind\":\"assistant\""), "persisted assistant") &&
         expect(has(body, "\"kind\":\"finish\""), "persisted finish") &&
         expect(has(body, "\"timestamp\":\""), "timestamp persisted");
}

bool chat_error_contract() {
  auto c = cfg();
  lkjai::HttpRequest req;
  req.method = "POST";
  req.path = "/api/chat";
  req.body = "{\"message\":\"hello\",\"run_id\":\"r2\","
             "\"visible_event_kinds\":[\"error\"]}";
  lkjai::NativeHttpResponse model;
  model.status = 503;
  model.body = "{\"error\":\"down\"}";
  auto resp = lkjai::runtime_chat_with_model_response(c, req, model);
  return expect(has(resp.body, "\"stop_reason\":\"model_error\""),
                "model error stop") &&
         expect(has(resp.body, "\"kind\":\"error\""), "error visible") &&
         expect(!has(resp.body, "\"kind\":\"user\""), "user filtered");
}

bool model_status_contract() {
  auto c = cfg();
  lkjai::NativeHttpResponse probe;
  probe.status = 200;
  probe.body = "{\"data\":[],\"device\":\"cuda\",\"cuda_available\":true,"
               "\"gpu_name\":\"NVIDIA\",\"warning\":\"\","
               "\"artifact_kind\":\"decoder\",\"chat_supported\":true,"
               "\"decode_supported\":true,\"degraded\":false}";
  auto body = lkjai::runtime_model_status_json(c, probe);
  return expect(has(body, "\"reachable\":true"), "reachable true") &&
         expect(has(body, "\"device\":\"cuda\""), "device field") &&
         expect(has(body, "\"cuda_available\":true"), "cuda field") &&
         expect(has(body, "\"artifact_kind\":\"decoder\""), "artifact kind") &&
         expect(has(body, "\"chat_supported\":true"), "chat supported") &&
         expect(has(body, "\"probe_status\":200"), "probe status");
}

bool degraded_model_status_contract() {
  auto c = cfg();
  lkjai::NativeHttpResponse probe;
  probe.status = 503;
  probe.body = "{\"loaded\":false,\"artifact_kind\":\"dense\","
               "\"chat_supported\":false,\"decode_supported\":false,"
               "\"degraded\":true,\"degraded_reason\":\"wrong kind\","
               "\"device\":\"cuda\",\"cuda_available\":true}";
  auto body = lkjai::runtime_model_status_json(c, probe);
  return expect(has(body, "\"reachable\":true"), "degraded reachable") &&
         expect(has(body, "\"loaded\":false"), "degraded unloaded") &&
         expect(has(body, "\"artifact_kind\":\"dense\""), "degraded kind") &&
         expect(has(body, "\"chat_supported\":false"), "degraded chat") &&
         expect(has(body, "\"degraded_reason\":\"wrong kind\""),
                "degraded reason");
}

bool config_status_contract() {
  auto c = cfg();
  c.kjxlkj_api_url = "http://kjxlkj.local";
  auto body = lkjai::runtime_config_status_json(c);
  return expect(has(body, "\"status\":\"degraded\""), "degraded status") &&
         expect(has(body, "\"local_only\":true"), "local bind") &&
	         expect(has(body, "/api/users/default/resources"), "resource base") &&
	         expect(has(body, "\"mutable_tools_enabled\":false"), "mutable tools") &&
         expect(has(body, "\"available\":[\"agent.finish\",\"agent.think\""),
                "tool registry");
}

bool health_contract() {
  auto body = lkjai::runtime_health_json(cfg());
  return expect(has(body, "\"status\":\"ok\""), "health status") &&
         expect(has(body, "\"service\":\"lkjai-native-runtime\""),
                "health service") &&
         expect(has(body, "\"host\":\"127.0.0.1\""), "health bind");
}

bool run_id_guard_contract() {
  auto c = cfg();
  lkjai::NativeHttpResponse model;
  model.status = 200;
  model.body =
      "{\"choices\":[{\"message\":{\"content\":\"<action><tool>agent.finish</tool><content>x</content></action>\"}}]}";
  lkjai::HttpRequest req{"POST", "/api/chat",
                         "{\"message\":\"hello\",\"run_id\":\"../bad\"}"};
  auto resp = lkjai::runtime_chat_with_model_response(c, req, model);
  auto run = lkjai::runtime_route(c, {"GET", "/api/runs/../bad", ""});
  return expect(resp.status == 400, "bad chat run id rejected") &&
         expect(run.status == 400, "bad route run id rejected");
}

bool route_boundary_contract() {
  auto c = cfg();
  auto health = lkjai::runtime_route(c, {"GET", "/healthz", ""});
  auto v1 = lkjai::runtime_route(c, {"GET", "/v1/models", ""});
  auto root = lkjai::runtime_route(c, {"GET", "/", ""});
  auto preflight = lkjai::runtime_route(c, {"OPTIONS", "/api/chat", ""});
  return expect(health.status == 200, "sandbox health") &&
         expect(v1.status == 404, "sandbox rejects v1") &&
         expect(root.status == 404, "sandbox rejects frontend") &&
         expect(preflight.status == 204, "sandbox preflight");
}

bool runs_list_contract() {
  auto c = cfg();
  auto empty = lkjai::runtime_route(c, {"GET", "/api/runs", ""});
  lkjai::runtime_append_event(c, "run-100", "user", "older prompt");
  lkjai::runtime_append_event(c, "run-100", "assistant", "older answer");
  lkjai::runtime_append_event(c, "run-200", "user", "newer prompt");
  lkjai::runtime_append_event(c, "run-200", "error", "newer failure");
  auto list = lkjai::runtime_route(c, {"GET", "/api/runs?limit=20", ""});
  auto one = lkjai::runtime_route(c, {"GET", "/api/runs?limit=1", ""});
  for (int i = 0; i < 105; ++i) {
    lkjai::runtime_append_event(c, "run-extra-" + std::to_string(i),
                                "user", "bulk");
  }
  auto clamped = lkjai::runtime_route(c, {"GET", "/api/runs?limit=500", ""});
  return expect(empty.status == 200, "runs empty status") &&
         expect(has(empty.body, "\"runs\":[]"), "runs empty body") &&
         expect(list.status == 200, "runs list status") &&
         expect(list.body.find("run-200") < list.body.find("run-100"),
                "runs newest first") &&
         expect(has(list.body, "\"event_count\":2"), "runs event count") &&
         expect(has(list.body, "\"last_kind\":\"error\""), "runs last kind") &&
         expect(has(list.body, "\"preview\":\"newer failure\""),
                "runs preview") &&
         expect(count_runs(one.body) == 1, "runs limit") &&
         expect(count_runs(clamped.body) == 100, "runs clamp");
}

}  // namespace

int main() {
	  return chat_filter_contract() && chat_error_contract() &&
	                 model_status_contract() &&
	                 degraded_model_status_contract() &&
	                 config_status_contract() &&
	                 health_contract() && run_id_guard_contract() &&
	                 route_boundary_contract() && runs_list_contract()
	             ? 0
	             : 1;
}
