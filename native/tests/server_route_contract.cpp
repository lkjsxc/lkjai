#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "json_contract.hpp"
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
  auto root_page = lkjai::native_server_route({"GET", "/", ""},
                                              dense, cuda, cfg);
  auto preflight = lkjai::native_server_route({"OPTIONS", "/v1/models", ""},
                                              dense, cuda, cfg);
  return expect(models.status == 200, "models status") &&
         expect(lkjai_test::valid_json(models.body), "models valid json") &&
         expect(has(models.body, "\"id\":\"dense-model\""), "models body") &&
         expect(has(models.body, "\"artifact_kind\":\"dense\""),
                "dense model kind") &&
         expect(has(models.body, "\"chat_supported\":false"),
                "dense chat unsupported metadata") &&
         expect(dense_chat.status == 422, "dense chat unsupported") &&
         expect(lkjai_test::valid_json(dense_chat.body), "chat valid json") &&
         expect(!has(dense_chat.body, "\"choices\""), "dense choices absent") &&
         expect(api_model.status == 404, "inference rejects api") &&
         expect(root_page.status == 404, "inference rejects frontend") &&
         expect(preflight.status == 204, "inference preflight");
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
         expect(lkjai_test::valid_json(models.body), "missing models json") &&
         expect(has(models.body, "missing artifact"), "missing error body") &&
         expect(chat.status == 503, "missing chat status") &&
         expect(lkjai_test::valid_json(chat.body), "missing chat json") &&
         expect(!has(chat.body, "\"choices\""), "missing chat choices absent") &&
         expect(health.status == 200, "health still ok") &&
         expect(lkjai_test::valid_json(health.body), "health valid json") &&
         expect(has(health.body, "\"loaded\":false"), "health loaded false") &&
         expect(has(health.body, "\"degraded\":true"), "health degraded");
}

}  // namespace

int main() {
  return route_contracts() && missing_model_contract() ? 0 : 1;
}
