#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "native_server_routes.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
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
  auto health = lkjai::native_server_route({"GET", "/healthz", ""},
                                           missing, cuda, cfg);
  return expect(models.status == 503, "missing models status") &&
         expect(has(models.body, "missing artifact"), "missing error body") &&
         expect(health.status == 200, "health still ok") &&
         expect(has(health.body, "\"loaded\":false"), "health loaded false");
}

}  // namespace

int main() {
  return route_contracts() && missing_model_contract() ? 0 : 1;
}
