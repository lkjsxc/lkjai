#include "cuda_probe.hpp"

#include <cuda_runtime.h>

namespace lkjai {

CudaStatus cuda_status() {
  CudaStatus status;
  int count = 0;
  auto result = cudaGetDeviceCount(&count);
  if (result != cudaSuccess || count <= 0) {
    status.warning = "CUDA unavailable";
    return status;
  }
  cudaDeviceProp prop{};
  result = cudaGetDeviceProperties(&prop, 0);
  if (result != cudaSuccess) {
    status.warning = "CUDA device properties unavailable";
    return status;
  }
  status.available = true;
  status.device = prop.name;
  status.compute_major = prop.major;
  status.compute_minor = prop.minor;
  status.bf16_supported = prop.major >= 8;
  return status;
}

}  // namespace lkjai
