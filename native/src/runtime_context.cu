#include "runtime_device.hpp"

namespace lkjai {

CudaExecutionContext::CudaExecutionContext() {
  require_cuda(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking),
               "cudaStreamCreate compute");
  require_cuda(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
               "cudaStreamCreate copy");
  require_cublaslt(cublasLtCreate(&cublaslt_), "cublasLtCreate");
  require_cudnn(cudnnCreate(&cudnn_), "cudnnCreate");
  require_cudnn(cudnnSetStream(cudnn_, compute_stream_), "cudnnSetStream");
}

CudaExecutionContext::~CudaExecutionContext() {
  if (cudnn_) cudnnDestroy(cudnn_);
  if (cublaslt_) cublasLtDestroy(cublaslt_);
  if (copy_stream_) cudaStreamDestroy(copy_stream_);
  if (compute_stream_) cudaStreamDestroy(compute_stream_);
}

}  // namespace lkjai
