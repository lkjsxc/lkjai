#include "cuda_probe.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace lkjai {
namespace {

bool can_create_cublaslt(std::string* error) {
  cublasLtHandle_t handle{};
  auto status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    *error = "cublasLtCreate failed";
    return false;
  }
  cublasLtDestroy(handle);
  return true;
}

bool can_create_cudnn(std::string* error) {
  cudnnHandle_t handle{};
  auto status = cudnnCreate(&handle);
  if (status != CUDNN_STATUS_SUCCESS) {
    *error = cudnnGetErrorString(status);
    return false;
  }
  cudnnDestroy(handle);
  return true;
}

}  // namespace

CudaStatus cuda_status() {
  CudaStatus status;
  cudaDriverGetVersion(&status.cuda_driver_version);
  cudaRuntimeGetVersion(&status.cuda_runtime_version);
  status.cudnn_version = static_cast<long long>(cudnnGetVersion());
  int count = 0;
  auto result = cudaGetDeviceCount(&count);
  status.device_count = count;
  if (result != cudaSuccess || count <= 0) {
    status.warning = "CUDA unavailable";
    return status;
  }
  if (cudaGetDevice(&status.device_index) != cudaSuccess) {
    status.device_index = 0;
  }
  cudaDeviceProp prop{};
  result = cudaGetDeviceProperties(&prop, status.device_index);
  if (result != cudaSuccess) {
    status.warning = "CUDA device properties unavailable";
    return status;
  }
  status.available = true;
  status.device = prop.name;
  status.compute_major = prop.major;
  status.compute_minor = prop.minor;
  status.total_global_memory = static_cast<size_t>(prop.totalGlobalMem);
  status.sm_count = prop.multiProcessorCount;
  status.bf16_supported = prop.major >= 8;
  status.cublaslt_available = can_create_cublaslt(&status.error);
  status.cudnn_available = can_create_cudnn(&status.error);
  int supported = 0;
  if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported,
                             status.device_index) == cudaSuccess) {
    status.async_alloc_supported = supported != 0;
  }
  status.sdpa_eligible = status.bf16_supported && status.cudnn_available;
  if (!cuda_required_ok(status)) {
    status.warning = status.error.empty() ? "CUDA capability incomplete"
                                          : status.error;
  }
  return status;
}

bool cuda_required_ok(const CudaStatus& status) {
  return status.available && status.bf16_supported &&
         status.cublaslt_available && status.cudnn_available;
}

}  // namespace lkjai
