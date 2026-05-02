#include "runtime_device.hpp"

namespace lkjai {

DeviceWorkspace::DeviceWorkspace(cudaStream_t stream) : stream_(stream) {
  int supported = 0;
  auto status = cudaDeviceGetAttribute(&supported,
                                       cudaDevAttrMemoryPoolsSupported, 0);
  async_supported_ = status == cudaSuccess && supported != 0;
}

DeviceWorkspace::~DeviceWorkspace() { reset(); }

void* DeviceWorkspace::allocate(size_t bytes) {
  reset();
  if (bytes == 0) return nullptr;
  bytes_reserved_ = bytes;
  if (async_supported_) {
    require_cuda(cudaMallocAsync(&data_, bytes, stream_), "cudaMallocAsync");
  } else {
    require_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
  }
  return data_;
}

void DeviceWorkspace::reset() {
  if (!data_) return;
  if (async_supported_) {
    cudaFreeAsync(data_, stream_);
    cudaStreamSynchronize(stream_);
  } else {
    cudaFree(data_);
  }
  data_ = nullptr;
  bytes_reserved_ = 0;
}

}  // namespace lkjai
