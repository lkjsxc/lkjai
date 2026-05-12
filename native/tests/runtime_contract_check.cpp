#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "runtime_api.hpp"

namespace {

bool has(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
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
               "\"gpu_name\":\"NVIDIA\",\"warning\":\"\"}";
  auto body = lkjai::runtime_model_status_json(c, probe);
  return expect(has(body, "\"reachable\":true"), "reachable true") &&
         expect(has(body, "\"device\":\"cuda\""), "device field") &&
         expect(has(body, "\"cuda_available\":true"), "cuda field") &&
         expect(has(body, "\"probe_status\":200"), "probe status");
}

bool config_status_contract() {
  auto c = cfg();
  c.kjxlkj_api_url = "http://kjxlkj.local";
  auto body = lkjai::runtime_config_status_json(c);
  return expect(has(body, "\"status\":\"degraded\""), "degraded status") &&
         expect(has(body, "\"local_only\":true"), "local bind") &&
	         expect(has(body, "/api/users/default/resources"), "resource base") &&
	         expect(has(body, "\"mutable_tools_enabled\":false"), "mutable tools");
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

}  // namespace

int main() {
	  return chat_filter_contract() && chat_error_contract() &&
	                 model_status_contract() && config_status_contract() &&
	                 health_contract() && run_id_guard_contract()
	             ? 0
	             : 1;
}
