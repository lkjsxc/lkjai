#include "runtime_device.hpp"

#include <stdexcept>
#include <string>

namespace lkjai {

void require_cuda(cudaError_t status, const char* label) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(label) + ": " +
                             cudaGetErrorString(status));
  }
}

void require_cublaslt(cublasStatus_t status, const char* label) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(label) + ": cuBLASLt failure");
  }
}

void require_cudnn(cudnnStatus_t status, const char* label) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(label) + ": " +
                             cudnnGetErrorString(status));
  }
}

}  // namespace lkjai
