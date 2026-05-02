#pragma once

#include <cstddef>
#include <cstdint>
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
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  DeviceTensor(DeviceTensor&& other) noexcept;
  DeviceTensor& operator=(DeviceTensor&& other) noexcept;
  ~DeviceTensor();

  void reset();
  void copy_from_host_f32(const std::vector<float>& host);
  std::vector<float> copy_to_host_f32() const;

  void* data() const { return data_; }
  const DeviceTensorSpec& spec() const { return spec_; }
  size_t bytes() const { return spec_.bytes(); }

 private:
  DeviceTensorSpec spec_;
  void* data_ = nullptr;
};

class CudaExecutionContext {
 public:
  CudaExecutionContext();
  CudaExecutionContext(const CudaExecutionContext&) = delete;
  CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
  ~CudaExecutionContext();

  cudaStream_t stream() const { return stream_; }
  cublasLtHandle_t cublaslt() const { return cublaslt_; }
  cudnnHandle_t cudnn() const { return cudnn_; }

 private:
  cudaStream_t stream_ = nullptr;
  cublasLtHandle_t cublaslt_ = nullptr;
  cudnnHandle_t cudnn_ = nullptr;
};

const char* dtype_name(DeviceDType dtype);

}  // namespace lkjai
