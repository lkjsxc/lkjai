#pragma once

#include <cstddef>
#include <string>

namespace lkjai {

struct CudaStatus {
  bool available = false;
  std::string device;
  int compute_major = 0;
  int compute_minor = 0;
  int cuda_driver_version = 0;
  int cuda_runtime_version = 0;
  long long cudnn_version = 0;
  int device_count = 0;
  int device_index = 0;
  size_t total_global_memory = 0;
  int sm_count = 0;
  bool bf16_supported = false;
  bool cublaslt_available = false;
  bool cudnn_available = false;
  bool sdpa_eligible = false;
  bool sdpa_shape_available = false;
  bool sdpa_shape_forward_supported = false;
  bool sdpa_shape_backward_supported = false;
  bool sdpa_shape_decode_supported = false;
  int sdpa_shape_head_dim = 72;
  std::string sdpa_shape_layout = "BSHD";
  std::string sdpa_shape_reason = "not_probed";
  bool async_alloc_supported = false;
  std::string warning;
  std::string error;
};

CudaStatus cuda_status();
bool cuda_required_ok(const CudaStatus& status);

}  // namespace lkjai
