#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "dense_train.hpp"
#include "packed_cache.hpp"

namespace lkjai {

struct DenseCudaCheck {
  bool ok = false;
  std::string device;
  int compute_major = 0;
  int compute_minor = 0;
  int cuda_runtime_version = 0;
  long long cudnn_version = 0;
  bool bf16_supported = false;
  bool cublaslt_available = false;
  bool cudnn_available = false;
  bool sdpa_eligible = false;
  bool async_alloc_supported = false;
  double loss = 0.0;
  double cpu_loss = 0.0;
  double max_logit_diff = 0.0;
  double max_grad_diff = 0.0;
  double max_update_diff = 0.0;
  std::string error;
};

DenseCudaCheck run_dense_cuda_check();
bool run_dense_cuda_training(const DenseTrainOptions& opt,
                             DenseTrainReport* report, std::string* error);
bool dense_cuda_logits_check(const std::filesystem::path& model_dir,
                             const std::string& token_csv, std::string* json,
                             std::string* error);
bool dense_cuda_logits_check_against_checkpoint(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& reference_checkpoint,
    const std::string& token_csv, std::string* json, std::string* error);

}  // namespace lkjai
