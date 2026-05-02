#include "transformer_state.hpp"

#include <cuda_runtime.h>

namespace lkjai {
namespace {

__global__ void transformer_probe_kernel(float* value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *value = *value + 1.0f;
}

}  // namespace

bool transformer_cuda_step_probe(std::string* error) {
  float* device = nullptr;
  if (cudaMalloc(&device, sizeof(float)) != cudaSuccess) {
    *error = "failed to allocate transformer CUDA probe buffer";
    return false;
  }
  float host = 0.0f;
  cudaMemcpy(device, &host, sizeof(float), cudaMemcpyHostToDevice);
  transformer_probe_kernel<<<1, 32>>>(device);
  auto status = cudaDeviceSynchronize();
  cudaMemcpy(&host, device, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (status != cudaSuccess || host != 1.0f) {
    *error = "transformer CUDA probe kernel failed";
    return false;
  }
  return true;
}

}  // namespace lkjai
