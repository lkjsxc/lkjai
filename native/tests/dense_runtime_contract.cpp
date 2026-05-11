#include <filesystem>
#include <fstream>
#include <iostream>

#include "dense_demo.hpp"
#include "dense_train_internal.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const char* message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

lkjai::ArtifactStatus dense_artifact(const std::filesystem::path& root) {
  auto dir = root / "models" / "dense-real";
  lkjai::DenseConfig cfg;
  cfg.model = "dense-real";
  cfg.vocab_size = 16;
  cfg.context = 8;
  cfg.hidden_size = 8;
  cfg.head_dim = 8;
  lkjai::DenseTrainState state;
  lkjai::init_dense_state(cfg, &state);
  std::string checksum;
  lkjai::write_dense_train_artifact(dir, state, 7, 14, 1, 3, 2, 1.25, false,
                                    &checksum);
  lkjai::ArtifactStatus artifact;
  artifact.loaded = true;
  artifact.model_name = "dense-real";
  artifact.model_dir = dir;
  return artifact;
}

}  // namespace

int main() {
  auto root = std::filesystem::path("/tmp/lkjai-dense-runtime-contract");
  std::filesystem::remove_all(root);
  auto artifact = dense_artifact(root);
  std::filesystem::create_directories(root / "data" / "runs");
  std::ofstream(root / "data" / "runs" / "train-report.json")
      << "{\"schema\":\"lkjai-train-report\",\"optimizer_steps\":7}\n";
  auto runtime = lkjai::load_dense_demo_runtime(artifact, root / "data");
  std::filesystem::remove(artifact.model_dir / "weights.lkjw");
  auto status = lkjai::dense_demo_status_response(runtime);
  auto next = lkjai::dense_demo_next_token_response(
      runtime, {"POST", "/api/dense/next-token", "{\"tokens\":[1,2],\"top_k\":3}"});
  bool ok = expect(status.status == 200, "status code") &&
            expect(has(status.body, "\"optimizer_steps\":7"), "steps") &&
            expect(has(status.body, "\"parameter_count\":256"), "params") &&
            expect(has(status.body, "train-report.json"), "provenance") &&
            expect(next.status == 200, "next code") &&
            expect(has(next.body, "\"top_k\":["), "top k") &&
            expect(has(next.body, "\"config_checksum\":\""), "config checksum");
  return ok ? 0 : 1;
}
