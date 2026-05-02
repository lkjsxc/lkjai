#include "runtime_device.hpp"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace lkjai {
namespace {

__global__ void f32_to_bf16(const float* in, __nv_bfloat16* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    reinterpret_cast<uint16_t*>(out)[i] =
        static_cast<uint16_t>((__float_as_uint(in[i]) + 0x8000u) >> 16);
  }
}
__global__ void bf16_to_f32(const __nv_bfloat16* in, float* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}
size_t dtype_size(DeviceDType dtype) {
  return dtype == DeviceDType::f32 ? sizeof(float) : sizeof(__nv_bfloat16);
}
bool async_alloc_supported() {
  int supported = 0;
  auto status = cudaDeviceGetAttribute(&supported,
                                       cudaDevAttrMemoryPoolsSupported, 0);
  return status == cudaSuccess && supported != 0;
}
void* allocate_temp(size_t bytes, cudaStream_t stream, bool* async) {
  void* ptr = nullptr;
  *async = async_alloc_supported();
  if (*async) {
    require_cuda(cudaMallocAsync(&ptr, bytes, stream), "cudaMallocAsync temp");
  } else {
    require_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc temp");
  }
  return ptr;
}

void free_temp(void* ptr, cudaStream_t stream, bool async) {
  if (async) {
    require_cuda(cudaFreeAsync(ptr, stream), "cudaFreeAsync temp");
  } else {
    require_cuda(cudaStreamSynchronize(stream), "temp sync");
    cudaFree(ptr);
  }
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
  if (spec_.bytes() > 0) {
    require_cuda(cudaMalloc(&data_, spec_.bytes()), "cudaMalloc");
  }
}

DeviceTensor::DeviceTensor(DeviceTensorSpec spec, cudaStream_t stream)
    : spec_(std::move(spec)) {
  if (spec_.bytes() > 0) {
    if (async_alloc_supported()) {
      require_cuda(cudaMallocAsync(&data_, spec_.bytes(), stream),
                   "cudaMallocAsync");
      alloc_stream_ = stream;
      async_alloc_ = true;
    } else {
      require_cuda(cudaMalloc(&data_, spec_.bytes()), "cudaMalloc");
    }
  }
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept {
  spec_ = std::move(other.spec_);
  data_ = other.data_;
  alloc_stream_ = other.alloc_stream_;
  async_alloc_ = other.async_alloc_;
  other.data_ = nullptr;
  other.async_alloc_ = false;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
  if (this != &other) {
    reset();
    spec_ = std::move(other.spec_);
    data_ = other.data_;
    alloc_stream_ = other.alloc_stream_;
    async_alloc_ = other.async_alloc_;
    other.data_ = nullptr;
    other.async_alloc_ = false;
  }
  return *this;
}

DeviceTensor::~DeviceTensor() { reset(); }

void DeviceTensor::reset() {
  if (data_ && async_alloc_) {
    cudaFreeAsync(data_, alloc_stream_);
    cudaStreamSynchronize(alloc_stream_);
  } else if (data_) {
    cudaFree(data_);
  }
  data_ = nullptr;
  alloc_stream_ = nullptr;
  async_alloc_ = false;
  spec_ = {};
}

void DeviceTensor::copy_from_host_f32(const std::vector<float>& host) {
  copy_from_host_f32(host, nullptr);
  require_cuda(cudaDeviceSynchronize(), "DeviceTensor copy_from_host sync");
}

void DeviceTensor::copy_from_host_f32(const std::vector<float>& host,
                                      cudaStream_t stream) {
  if (host.size() != spec_.elements()) {
    throw std::runtime_error("host element count does not match tensor shape");
  }
  if (spec_.dtype == DeviceDType::f32) {
    require_cuda(cudaMemcpyAsync(data_, host.data(), spec_.bytes(),
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync H2D f32");
    return;
  }
  float* temp = nullptr;
  bool async = false;
  temp = static_cast<float*>(allocate_temp(host.size() * sizeof(float), stream,
                                          &async));
  require_cuda(cudaMemcpyAsync(temp, host.data(), host.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync H2D temp");
  f32_to_bf16<<<static_cast<unsigned>((host.size() + 255) / 256), 256, 0,
                stream>>>(
      temp, static_cast<__nv_bfloat16*>(data_), host.size());
  require_cuda(cudaGetLastError(), "f32_to_bf16");
  free_temp(temp, stream, async);
}

std::vector<float> DeviceTensor::copy_to_host_f32() const {
  return copy_to_host_f32(nullptr);
}

std::vector<float> DeviceTensor::copy_to_host_f32(cudaStream_t stream) const {
  std::vector<float> host(spec_.elements());
  if (host.empty()) return host;
  if (spec_.dtype == DeviceDType::f32) {
    require_cuda(cudaMemcpyAsync(host.data(), data_, spec_.bytes(),
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync D2H f32");
    require_cuda(cudaStreamSynchronize(stream), "D2H f32 sync");
    return host;
  }
  float* temp = nullptr;
  bool async = false;
  temp = static_cast<float*>(allocate_temp(host.size() * sizeof(float), stream,
                                          &async));
  bf16_to_f32<<<static_cast<unsigned>((host.size() + 255) / 256), 256, 0,
                stream>>>(
      static_cast<const __nv_bfloat16*>(data_), temp, host.size());
  require_cuda(cudaGetLastError(), "bf16_to_f32");
  require_cuda(cudaMemcpyAsync(host.data(), temp, host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync D2H temp");
  free_temp(temp, stream, async);
  require_cuda(cudaStreamSynchronize(stream), "D2H bf16 sync");
  return host;
}

CudaExecutionContext::CudaExecutionContext() {
  require_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  require_cublaslt(cublasLtCreate(&cublaslt_), "cublasLtCreate");
  require_cudnn(cudnnCreate(&cudnn_), "cudnnCreate");
  require_cudnn(cudnnSetStream(cudnn_, stream_), "cudnnSetStream");
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
