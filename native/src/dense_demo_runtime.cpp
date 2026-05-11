#include "dense_demo.hpp"

#include "dense_cuda_internal.hpp"
#include "dense_report_util.hpp"
#include "json_min.hpp"
#include "train_report_digest.hpp"

namespace lkjai {
namespace {

std::filesystem::path report_path(const std::filesystem::path& data_dir) {
  if (data_dir.empty()) return {};
  return data_dir / "runs" / "train-report.json";
}

}  // namespace

DenseDemoRuntime load_dense_demo_runtime(const ArtifactStatus& artifact,
                                         const std::filesystem::path& data_dir) {
  DenseDemoRuntime runtime;
  runtime.artifact = artifact;
  if (!artifact.loaded) {
    runtime.error = artifact.error;
    return runtime;
  }
  runtime.manifest = read_text(artifact.model_dir / "manifest.json");
  runtime.dense_supported = contains_json_string(runtime.manifest, "kind", "dense");
  if (!runtime.dense_supported) {
    runtime.error = "loaded artifact does not support dense logits";
    return runtime;
  }
  runtime.config = dense_config_from_artifact(artifact.model_dir);
  runtime.weights_checksum = json_first_string(runtime.manifest, "weights_checksum");
  runtime.config_checksum = json_first_string(runtime.manifest, "config_checksum");
  runtime.trainer_state = read_text(artifact.model_dir / "trainer_state.json");
  runtime.optimizer_steps =
      json_int_value(runtime.trainer_state, "optimizer_steps", 0);
  runtime.loss = json_double_value(runtime.trainer_state, "loss", 0.0);
  runtime.parameter_count = static_cast<long long>(runtime.config.vocab_size) *
                            runtime.config.hidden_size * 2;
  std::string error;
  runtime.embeddings =
      read_dense_tensor(artifact.model_dir, "tok_embeddings", &error);
  runtime.head = read_dense_tensor(artifact.model_dir, "lm_head", &error);
  if (!error.empty()) {
    runtime.dense_supported = false;
    runtime.error = error;
  }
  auto path = report_path(data_dir);
  if (!path.empty() && std::filesystem::is_regular_file(path)) {
    runtime.train_report_path = path.string();
    runtime.train_report_digest = train_report_file_digest(path);
  }
  return runtime;
}

}  // namespace lkjai
