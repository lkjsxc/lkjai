#pragma once

#include <filesystem>
#include <vector>

#include "artifact.hpp"
#include "dense_train.hpp"
#include "http_server.hpp"

namespace lkjai {

struct DenseDemoRuntime {
  ArtifactStatus artifact;
  bool dense_supported = false;
  std::string error;
  std::string manifest;
  std::string trainer_state;
  std::string train_report_path;
  std::string train_report_digest;
  std::string weights_checksum;
  std::string config_checksum;
  int optimizer_steps = 0;
  double loss = 0.0;
  long long parameter_count = 0;
  DenseConfig config;
  std::vector<float> embeddings;
  std::vector<float> head;
};

DenseDemoRuntime load_dense_demo_runtime(const ArtifactStatus& artifact,
                                         const std::filesystem::path& data_dir);
HttpResponse dense_demo_status_response(const ArtifactStatus& artifact);
HttpResponse dense_demo_status_response(const DenseDemoRuntime& runtime);
HttpResponse dense_demo_next_token_response(const ArtifactStatus& artifact,
                                            const HttpRequest& request);
HttpResponse dense_demo_next_token_response(const DenseDemoRuntime& runtime,
                                            const HttpRequest& request);

}  // namespace lkjai
