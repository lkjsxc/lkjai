#include "runtime_device.hpp"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace lkjai {
namespace {

void require(cudaError_t status, const char* label) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(label) + ": " +
                             cudaGetErrorString(status));
  }
}

void require(cublasStatus_t status, const char* label) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(label) + ": cuBLASLt failure");
  }
}

void require(cudnnStatus_t status, const char* label) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(label) + ": " +
                             cudnnGetErrorString(status));
  }
}

__global__ void f32_to_bf16(const float* in, __nv_bfloat16* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ void bf16_to_f32(const __nv_bfloat16* in, float* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

size_t dtype_size(DeviceDType dtype) {
  return dtype == DeviceDType::f32 ? sizeof(float) : sizeof(__nv_bfloat16);
}

}  // namespace

size_t DeviceTensorSpec::elements() const {
  size_t total = 1;
  for (auto dim : shape) {
    if (dim < 1) return 0;
    total *= static_cast<size_t>(dim);
  }
  return shape.empty() ? 0 : total;
}

size_t DeviceTensorSpec::bytes() const { return elements() * dtype_size(dtype); }

DeviceTensor::DeviceTensor(DeviceTensorSpec spec) : spec_(std::move(spec)) {
  if (spec_.bytes() > 0) require(cudaMalloc(&data_, spec_.bytes()), "cudaMalloc");
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept {
  spec_ = std::move(other.spec_);
  data_ = other.data_;
  other.data_ = nullptr;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
  if (this != &other) {
    reset();
    spec_ = std::move(other.spec_);
    data_ = other.data_;
    other.data_ = nullptr;
  }
  return *this;
}

DeviceTensor::~DeviceTensor() { reset(); }

void DeviceTensor::reset() {
  if (data_) cudaFree(data_);
  data_ = nullptr;
  spec_ = {};
}

void DeviceTensor::copy_from_host_f32(const std::vector<float>& host) {
  if (host.size() != spec_.elements()) {
    throw std::runtime_error("host element count does not match tensor shape");
  }
  if (spec_.dtype == DeviceDType::f32) {
    require(cudaMemcpy(data_, host.data(), spec_.bytes(), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D f32");
    return;
  }
  float* temp = nullptr;
  require(cudaMalloc(&temp, host.size() * sizeof(float)), "cudaMalloc temp");
  require(cudaMemcpy(temp, host.data(), host.size() * sizeof(float),
                     cudaMemcpyHostToDevice), "cudaMemcpy H2D temp");
  f32_to_bf16<<<static_cast<unsigned>((host.size() + 255) / 256), 256>>>(
      temp, static_cast<__nv_bfloat16*>(data_), host.size());
  require(cudaGetLastError(), "f32_to_bf16");
  require(cudaDeviceSynchronize(), "f32_to_bf16 sync");
  cudaFree(temp);
}

std::vector<float> DeviceTensor::copy_to_host_f32() const {
  std::vector<float> host(spec_.elements());
  if (host.empty()) return host;
  if (spec_.dtype == DeviceDType::f32) {
    require(cudaMemcpy(host.data(), data_, spec_.bytes(), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H f32");
    return host;
  }
  float* temp = nullptr;
  require(cudaMalloc(&temp, host.size() * sizeof(float)), "cudaMalloc temp");
  bf16_to_f32<<<static_cast<unsigned>((host.size() + 255) / 256), 256>>>(
      static_cast<const __nv_bfloat16*>(data_), temp, host.size());
  require(cudaGetLastError(), "bf16_to_f32");
  require(cudaMemcpy(host.data(), temp, host.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "cudaMemcpy D2H temp");
  cudaFree(temp);
  return host;
}

CudaExecutionContext::CudaExecutionContext() {
  require(cudaStreamCreate(&stream_), "cudaStreamCreate");
  require(cublasLtCreate(&cublaslt_), "cublasLtCreate");
  require(cudnnCreate(&cudnn_), "cudnnCreate");
  require(cudnnSetStream(cudnn_, stream_), "cudnnSetStream");
}

CudaExecutionContext::~CudaExecutionContext() {
  if (cudnn_) cudnnDestroy(cudnn_);
  if (cublaslt_) cublasLtDestroy(cublaslt_);
  if (stream_) cudaStreamDestroy(stream_);
}

const char* dtype_name(DeviceDType dtype) {
  return dtype == DeviceDType::f32 ? "f32" : "bf16";
}

}  // namespace lkjai
