#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace lkjai {

enum class DeviceDType { f32, bf16 };

struct DeviceTensorSpec {
  DeviceDType dtype = DeviceDType::f32;
  std::vector<int64_t> shape;

  size_t elements() const;
  size_t bytes() const;
};

class DeviceTensor {
 public:
  DeviceTensor() = default;
  explicit DeviceTensor(DeviceTensorSpec spec);
  DeviceTensor(DeviceTensorSpec spec, cudaStream_t stream);
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  DeviceTensor(DeviceTensor&& other) noexcept;
  DeviceTensor& operator=(DeviceTensor&& other) noexcept;
  ~DeviceTensor();

  void reset();
  void copy_from_host_f32(const std::vector<float>& host);
  void copy_from_host_f32(const std::vector<float>& host, cudaStream_t stream);
  std::vector<float> copy_to_host_f32() const;
  std::vector<float> copy_to_host_f32(cudaStream_t stream) const;

  void* data() const { return data_; }
  const DeviceTensorSpec& spec() const { return spec_; }
  size_t bytes() const { return spec_.bytes(); }

 private:
  DeviceTensorSpec spec_;
  void* data_ = nullptr;
  cudaStream_t alloc_stream_ = nullptr;
  bool async_alloc_ = false;
};

class DeviceWorkspace {
 public:
  explicit DeviceWorkspace(cudaStream_t stream);
  DeviceWorkspace(const DeviceWorkspace&) = delete;
  DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
  ~DeviceWorkspace();

  void* allocate(size_t bytes);
  void reset();
  bool async_supported() const { return async_supported_; }
  const std::string& backend() const { return backend_; }
  size_t bytes_reserved() const { return bytes_reserved_; }
  size_t high_water_bytes() const { return high_water_bytes_; }
  int reallocations() const { return reallocations_; }
  uint64_t release_threshold_bytes() const { return release_threshold_bytes_; }

 private:
  cudaStream_t stream_ = nullptr;
  void* data_ = nullptr;
  size_t bytes_reserved_ = 0;
  size_t high_water_bytes_ = 0;
  int reallocations_ = 0;
  uint64_t release_threshold_bytes_ = 0;
  bool async_supported_ = false;
  std::string backend_ = "legacy";
};

class CudaExecutionContext {
 public:
  CudaExecutionContext();
  CudaExecutionContext(const CudaExecutionContext&) = delete;
  CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
  ~CudaExecutionContext();

  cudaStream_t stream() const { return compute_stream_; }
  cudaStream_t compute_stream() const { return compute_stream_; }
  cudaStream_t copy_stream() const { return copy_stream_; }
  cublasLtHandle_t cublaslt() const { return cublaslt_; }
  cudnnHandle_t cudnn() const { return cudnn_; }

 private:
  cudaStream_t compute_stream_ = nullptr;
  cudaStream_t copy_stream_ = nullptr;
  cublasLtHandle_t cublaslt_ = nullptr;
  cudnnHandle_t cudnn_ = nullptr;
};

const char* dtype_name(DeviceDType dtype);
void require_cuda(cudaError_t status, const char* label);
void require_cublaslt(cublasStatus_t status, const char* label);
void require_cudnn(cudnnStatus_t status, const char* label);

}  // namespace lkjai
