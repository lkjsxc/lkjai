#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "native_server_routes.hpp"
#include "runtime_events.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
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

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

void write_manifest(const std::filesystem::path& dir, const std::string& kind) {
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "manifest.json")
      << "{\"format\":\"lkjai-native-artifact\",\"kind\":\"" << kind
      << "\",\"weights_checksum\":\"abc\"}\n";
}

lkjai::ArtifactStatus artifact(const std::filesystem::path& root,
                               const std::string& kind) {
  write_manifest(root / kind, kind);
  lkjai::ArtifactStatus a;
  a.loaded = true;
  a.model_name = kind + "-model";
  a.model_dir = root / kind;
  return a;
}

lkjai::RuntimeConfig runtime(const std::filesystem::path& root,
                             const std::string& model) {
  return {"127.0.0.1", 8080, (root / "data").string(),
          "local-native-engine", model, "readonly",
          (root / "workspace").string(), "http://127.0.0.1:9090",
          "default", ""};
}

bool route_contracts() {
  auto root = std::filesystem::path("/tmp/lkjai-server-route-contract");
  std::filesystem::remove_all(root);
  auto dense = artifact(root, "dense");
  lkjai::CudaStatus cuda;
  cuda.available = true;
  cuda.device = "contract-gpu";
  auto cfg = runtime(root, dense.model_name);
  auto models = lkjai::native_server_route({"GET", "/v1/models", ""},
                                           dense, cuda, cfg);
  auto dense_chat = lkjai::native_server_route(
      {"POST", "/v1/chat/completions", "{\"model\":\"dense-model\"}"},
      dense, cuda, cfg);
  auto api_model = lkjai::native_server_route({"GET", "/api/model", ""},
                                              dense, cuda, cfg);
  auto api_chat = lkjai::native_server_route(
      {"POST", "/api/chat", "{\"message\":\"hello\",\"run_id\":\"r1\"}"},
      dense, cuda, cfg);
  auto run = lkjai::native_server_route({"GET", "/api/runs/r1", ""},
                                        dense, cuda, cfg);
  return expect(models.status == 200, "models status") &&
         expect(has(models.body, "\"id\":\"dense-model\""), "models body") &&
         expect(dense_chat.status == 422, "dense chat unsupported") &&
         expect(!has(dense_chat.body, "\"choices\""), "dense choices absent") &&
         expect(api_model.status == 200, "api model status") &&
         expect(has(api_model.body, "\"reachable\":true"), "api reachable") &&
         expect(has(api_model.body, "\"artifact_kind\":\"dense\""),
                "api artifact kind") &&
         expect(has(api_model.body, "\"chat_supported\":false"),
                "api chat unsupported") &&
         expect(has(api_model.body, "\"dense_supported\":true"),
                "api dense supported") &&
         expect(has(api_model.body, "\"tool_profile\":\"readonly\""),
                "api tool profile") &&
         expect(api_chat.status == 200, "api chat status") &&
         expect(has(api_chat.body, "\"stop_reason\":\"model_error\""),
                "api chat model error") &&
         expect(run.status == 200, "run status") &&
         expect(has(run.body, "\"kind\":\"error\""), "run persisted error");
}

bool missing_model_contract() {
  lkjai::ArtifactStatus missing;
  missing.loaded = false;
  missing.error = "missing artifact";
  auto cfg = runtime("/tmp/lkjai-server-route-missing", "missing");
  lkjai::CudaStatus cuda;
  auto models = lkjai::native_server_route({"GET", "/v1/models", ""},
                                           missing, cuda, cfg);
  auto chat = lkjai::native_server_route(
      {"POST", "/v1/chat/completions",
       "{\"model\":\"missing\",\"messages\":[{\"role\":\"user\","
       "\"content\":\"hello\"}]}"},
      missing, cuda, cfg);
  auto health = lkjai::native_server_route({"GET", "/healthz", ""},
                                           missing, cuda, cfg);
  return expect(models.status == 503, "missing models status") &&
         expect(has(models.body, "missing artifact"), "missing error body") &&
         expect(chat.status == 503, "missing chat status") &&
         expect(!has(chat.body, "\"choices\""), "missing chat choices absent") &&
         expect(health.status == 200, "health still ok") &&
         expect(has(health.body, "\"loaded\":false"), "health loaded false");
}

bool runs_list_contract() {
  auto root = std::filesystem::path("/tmp/lkjai-server-route-runs");
  std::filesystem::remove_all(root);
  auto dense = artifact(root, "dense");
  lkjai::CudaStatus cuda;
  auto cfg = runtime(root, dense.model_name);
  auto empty = lkjai::native_server_route({"GET", "/api/runs", ""},
                                          dense, cuda, cfg);
  lkjai::runtime_append_event(cfg, "run-100", "user", "older prompt");
  lkjai::runtime_append_event(cfg, "run-100", "assistant", "older answer");
  lkjai::runtime_append_event(cfg, "run-200", "user", "newer prompt");
  lkjai::runtime_append_event(cfg, "run-200", "error", "newer failure");
  auto list = lkjai::native_server_route({"GET", "/api/runs?limit=20", ""},
                                         dense, cuda, cfg);
  auto one = lkjai::native_server_route({"GET", "/api/runs?limit=1", ""},
                                        dense, cuda, cfg);
  for (int i = 0; i < 105; ++i) {
    lkjai::runtime_append_event(cfg, "run-extra-" + std::to_string(i),
                                "user", "bulk");
  }
  auto clamped = lkjai::native_server_route({"GET", "/api/runs?limit=500", ""},
                                            dense, cuda, cfg);
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
  return route_contracts() && missing_model_contract() && runs_list_contract()
             ? 0
             : 1;
}
