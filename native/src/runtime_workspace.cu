#include "runtime_device.hpp"

#include <algorithm>

#include "dense_cuda_tuning.hpp"

namespace lkjai {
namespace {

uint64_t configure_pool_threshold(bool enabled) {
  if (!enabled) return 0;
  cudaMemPool_t pool{};
  if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess) return 0;
  uint64_t threshold = 256ull * 1024ull * 1024ull;
  auto status = cudaMemPoolSetAttribute(
      pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  return status == cudaSuccess ? threshold : 0;
}

}  // namespace

DeviceWorkspace::DeviceWorkspace(cudaStream_t stream) : stream_(stream) {
  int supported = 0;
  auto status = cudaDeviceGetAttribute(&supported,
                                       cudaDevAttrMemoryPoolsSupported, 0);
  bool supported_async = status == cudaSuccess && supported != 0;
  const auto& tuning = dense_runtime_tuning();
  async_supported_ = supported_async && tuning.allocator_mode != "legacy";
  backend_ = async_supported_ ? "cuda_malloc_async_pool" : "cuda_malloc";
  release_threshold_bytes_ = configure_pool_threshold(async_supported_);
}

DeviceWorkspace::~DeviceWorkspace() { reset(); }

void* DeviceWorkspace::allocate(size_t bytes) {
  if (data_ && bytes <= bytes_reserved_) return data_;
  reset();
  if (bytes == 0) return nullptr;
  bytes_reserved_ = bytes;
  high_water_bytes_ = std::max(high_water_bytes_, bytes_reserved_);
  ++reallocations_;
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
