#include "dense_cuda.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

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

bool ok(cublasStatus_t status, std::string* error, const char* label) {
  if (status == CUBLAS_STATUS_SUCCESS) return true;
  *error = std::string(label) + ": cuBLASLt failure";
  return false;
}

bool ok(cudnnStatus_t status, std::string* error, const char* label) {
  if (status == CUDNN_STATUS_SUCCESS) return true;
  *error = std::string(label) + ": " + cudnnGetErrorString(status);
  return false;
}

}  // namespace

DenseCudaCheck run_dense_cuda_check() {
  DenseCudaCheck check;
  int count = 0;
  if (!ok(cudaGetDeviceCount(&count), &check.error, "cudaGetDeviceCount")) {
    return check;
  }
  if (count <= 0) {
    check.error = "no CUDA devices";
    return check;
  }
  cudaDeviceProp prop{};
  if (!ok(cudaGetDeviceProperties(&prop, 0), &check.error,
          "cudaGetDeviceProperties")) {
    return check;
  }
  check.device = prop.name;
  check.compute_major = prop.major;
  check.compute_minor = prop.minor;
  cudaRuntimeGetVersion(&check.cuda_runtime_version);
  check.cudnn_version = static_cast<long long>(cudnnGetVersion());
  check.bf16_supported = prop.major >= 8;
  check.sdpa_eligible = check.bf16_supported && 72 % 8 == 0;
  if (!check.bf16_supported) {
    check.error = "BF16 dense path requires compute capability 8.0+";
    return check;
  }
  cublasLtHandle_t lt{};
  if (!ok(cublasLtCreate(&lt), &check.error, "cublasLtCreate")) return check;
  check.cublaslt_available = true;
  cublasLtDestroy(lt);
  cudnnHandle_t cudnn{};
  if (!ok(cudnnCreate(&cudnn), &check.error, "cudnnCreate")) return check;
  check.cudnn_available = true;
  cudnnDestroy(cudnn);
  float* device = nullptr;
  float host = 0.0f;
  if (!ok(cudaMalloc(&device, sizeof(float)), &check.error, "cudaMalloc")) {
    return check;
  }
  bf16_kernel<<<1, 1>>>(device);
  if (!ok(cudaGetLastError(), &check.error, "bf16_kernel")) return check;
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
