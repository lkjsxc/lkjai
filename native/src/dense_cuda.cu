#include "dense_cuda.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cuda_probe.hpp"

namespace lkjai {
namespace {

__global__ void bf16_kernel(float* out) {
  __nv_bfloat16 value = __float2bfloat16(1.25f);
  out[0] = __bfloat162float(value);
}

bool ok(cudaError_t status, std::string* error, const char* label) {
  if (status == cudaSuccess) return true;
  *error = std::string(label) + ": " + cudaGetErrorString(status);
  return false;
}

}  // namespace

DenseCudaCheck run_dense_cuda_check() {
  DenseCudaCheck check;
  auto status = cuda_status();
  check.device = status.device;
  check.compute_major = status.compute_major;
  check.compute_minor = status.compute_minor;
  check.cuda_runtime_version = status.cuda_runtime_version;
  check.cudnn_version = status.cudnn_version;
  check.bf16_supported = status.bf16_supported;
  check.cublaslt_available = status.cublaslt_available;
  check.cudnn_available = status.cudnn_available;
  check.sdpa_eligible = status.sdpa_eligible;
  check.async_alloc_supported = status.async_alloc_supported;
  check.error = status.error;
  if (!cuda_required_ok(status)) {
    if (check.error.empty()) check.error = status.warning;
    return check;
  }
  float* device = nullptr;
  float host = 0.0f;
  if (!ok(cudaMalloc(&device, sizeof(float)), &check.error, "cudaMalloc")) {
    return check;
  }
  bf16_kernel<<<1, 1>>>(device);
  if (!ok(cudaGetLastError(), &check.error, "bf16_kernel")) {
    cudaFree(device);
    return check;
  }
  if (!ok(cudaMemcpy(&host, device, sizeof(float), cudaMemcpyDeviceToHost),
          &check.error, "cudaMemcpy")) {
    cudaFree(device);
    return check;
  }
  cudaFree(device);
  check.ok = host > 1.24f && host < 1.26f;
  if (!check.ok) check.error = "BF16 roundtrip mismatch";
  return check;
}

}  // namespace lkjai
