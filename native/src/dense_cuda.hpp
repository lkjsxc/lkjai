#pragma once

#include <string>

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
  std::string error;
};

DenseCudaCheck run_dense_cuda_check();

}  // namespace lkjai
