#include <filesystem>
#include <iostream>
#include <string>

#include "dense_train_internal.hpp"
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

lkjai::ArtifactStatus dense_artifact(const std::filesystem::path& root) {
  auto dir = root / "dense-real";
  lkjai::DenseConfig cfg;
  cfg.model = "dense-real";
  cfg.vocab_size = 16;
  cfg.context = 8;
  cfg.hidden_size = 8;
  cfg.head_dim = 8;
  lkjai::DenseTrainState state;
  lkjai::init_dense_state(cfg, &state);
  std::string checksum;
  write_dense_train_artifact(dir, state, 2, 2, 1, 3, 1, 1.0, false,
                             &checksum);
  lkjai::ArtifactStatus a;
  a.loaded = true;
  a.model_name = "dense-real";
  a.model_dir = dir;
  return a;
}

lkjai::RuntimeConfig runtime(const std::filesystem::path& root,
                             const std::string& model) {
  return {"127.0.0.1", 8080, (root / "data").string(),
          "local-native-engine", model, "readonly",
          (root / "workspace").string(), "http://127.0.0.1:9090",
          "default", ""};
}

bool dense_demo_contract() {
  auto root = std::filesystem::path("/tmp/lkjai-server-route-dense");
  std::filesystem::remove_all(root);
  auto dense = dense_artifact(root);
  lkjai::CudaStatus cuda;
  auto cfg = runtime(root, dense.model_name);
  auto status = lkjai::native_server_route({"GET", "/api/dense/status", ""},
                                           dense, cuda, cfg);
  auto next = lkjai::native_server_route(
      {"POST", "/api/dense/next-token", "{\"tokens\":[1,2,3],\"top_k\":3}"},
      dense, cuda, cfg);
  auto bad = lkjai::native_server_route(
      {"POST", "/api/dense/next-token", "{\"tokens\":[],\"top_k\":3}"},
      dense, cuda, cfg);
  return expect(status.status == 200, "dense status code") &&
         expect(has(status.body, "\"dense_supported\":true"),
                "dense status supported") &&
         expect(next.status == 200, "dense next status") &&
         expect(has(next.body, "\"decode_supported\":false"),
                "dense decode unsupported") &&
         expect(has(next.body, "\"top_k\":["), "dense top k body") &&
         expect(has(next.body, "\"checksum\":\""), "dense checksum") &&
         expect(bad.status == 400, "dense bad request");
}

}  // namespace

int main() { return dense_demo_contract() ? 0 : 1; }
