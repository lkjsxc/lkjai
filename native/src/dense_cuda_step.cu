#include "dense_cuda_internal.hpp"

#include <algorithm>

#include <cuda_runtime.h>

namespace lkjai {
namespace {

class EventSpan {
 public:
  EventSpan(cudaStream_t stream, const char* label) : stream_(stream), label_(label) {
    require_cuda(cudaEventCreate(&start_), "cudaEventCreate start");
    require_cuda(cudaEventCreate(&stop_), "cudaEventCreate stop");
    require_cuda(cudaEventRecord(start_, stream_), label_);
  }
  ~EventSpan() {
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
  }
  double seconds() {
    require_cuda(cudaEventRecord(stop_, stream_), label_);
    require_cuda(cudaEventSynchronize(stop_), label_);
    float ms = 0.0f;
    require_cuda(cudaEventElapsedTime(&ms, start_, stop_), label_);
    return static_cast<double>(ms) / 1000.0;
  }

 private:
  cudaStream_t stream_ = nullptr;
  const char* label_ = "";
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace

void DenseCudaState::ensure_step_buffers(int rows) {
  if (rows <= step_rows_) return;
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  step_hidden_ = DeviceTensor({DeviceDType::bf16, {rows, h}}, ctx_->stream());
  step_out_ = DeviceTensor({DeviceDType::f32, {rows, v}}, ctx_->stream());
  step_grad_logits_ = DeviceTensor({DeviceDType::f32, {rows, v}}, ctx_->stream());
  step_d_hidden_ = DeviceTensor({DeviceDType::f32, {rows, h}}, ctx_->stream());
  step_loss_ = DeviceTensor({DeviceDType::f32, {1}}, ctx_->stream());
  step_rows_ = rows;
}

double DenseCudaState::forward_backward(const PackedBatch& batch,
                                        std::vector<float>* logits,
                                        double* h2d_seconds,
                                        double* fwd_seconds,
                                        double* bwd_seconds,
                                        float grad_scale,
                                        bool reset_grads) {
  int rows = batch.batch_size * batch.sequence_len;
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  ensure_batch_buffers(batch.tokens.size(), batch.loss_mask.size());
  ensure_step_buffers(rows);
  {
    EventSpan timer(ctx_->stream(), "dense H2D event");
    std::copy(batch.tokens.begin(), batch.tokens.end(), host_tokens_);
    std::copy(batch.loss_mask.begin(), batch.loss_mask.end(), host_mask_);
    require_cuda(cudaMemcpyAsync(device_tokens_, host_tokens_,
                                 batch.tokens.size() * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, ctx_->stream()),
                 "tokens H2D");
    require_cuda(cudaMemcpyAsync(device_mask_, host_mask_,
                                 batch.loss_mask.size(), cudaMemcpyHostToDevice,
                                 ctx_->stream()),
                 "mask H2D");
    if (h2d_seconds) *h2d_seconds += timer.seconds();
  }
  require_cuda(cudaMemsetAsync(step_loss_.data(), 0, sizeof(float),
                               ctx_->stream()), "loss memset");
  if (reset_grads) {
    require_cuda(cudaMemsetAsync(grad_emb_.data(), 0, grad_emb_.bytes(),
                                 ctx_->stream()), "grad emb memset");
    require_cuda(cudaMemsetAsync(grad_head_.data(), 0, grad_head_.bytes(),
                                 ctx_->stream()), "grad head memset");
  }
  {
    EventSpan timer(ctx_->stream(), "dense forward event");
    dense_launch_gather(device_tokens_, emb_shadow_.data(), step_hidden_.data(),
                        batch.batch_size, batch.sequence_len, v, h,
                        ctx_->stream());
    gemm(step_hidden_, step_out_, rows);
    dense_launch_loss_grad(static_cast<float*>(step_out_.data()),
                           device_tokens_, device_mask_,
                           static_cast<float*>(step_grad_logits_.data()),
                           static_cast<float*>(step_loss_.data()),
                           batch.batch_size, batch.sequence_len, v,
                           dense_supervised_count(batch), grad_scale,
                           ctx_->stream());
    if (fwd_seconds) *fwd_seconds += timer.seconds();
  }
  {
    EventSpan timer(ctx_->stream(), "dense backward event");
    gemm_head_grad(step_grad_logits_, step_hidden_, rows);
    gemm_d_hidden(step_grad_logits_, step_d_hidden_, rows);
    dense_launch_scatter_emb_grad(device_tokens_,
                                  static_cast<float*>(step_d_hidden_.data()),
                                  static_cast<float*>(grad_emb_.data()),
                                  batch.batch_size, batch.sequence_len, v, h,
                                  ctx_->stream());
    if (bwd_seconds) *bwd_seconds += timer.seconds();
  }
  auto loss_host = step_loss_.copy_to_host_f32(ctx_->stream());
  if (logits) {
    auto all = step_out_.copy_to_host_f32(ctx_->stream());
    auto row = static_cast<size_t>(rows - 2) * static_cast<size_t>(v);
    logits->assign(all.begin() + static_cast<std::ptrdiff_t>(row),
                   all.begin() + static_cast<std::ptrdiff_t>(row + v));
  }
  return loss_host.empty() ? 0.0 : loss_host[0];
}

}  // namespace lkjai
