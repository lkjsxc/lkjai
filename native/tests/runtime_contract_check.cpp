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
  model.body = "{\"choices\":[{\"message\":{\"content\":\"hi\"}}]}";
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

}  // namespace

int main() {
  return chat_filter_contract() && chat_error_contract() &&
                 model_status_contract()
             ? 0
             : 1;
}
