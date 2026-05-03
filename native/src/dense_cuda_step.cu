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
  step_hidden_f32_ = DeviceTensor({DeviceDType::f32, {rows, h}}, ctx_->stream());
  step_out_ = DeviceTensor({DeviceDType::f32, {rows, v}}, ctx_->stream());
  step_grad_logits_ = DeviceTensor({DeviceDType::f32, {rows, v}}, ctx_->stream());
  step_d_hidden_ = DeviceTensor({DeviceDType::f32, {rows, h}}, ctx_->stream());
  step_head_f32_ = DeviceTensor({DeviceDType::f32, {v, h}}, ctx_->stream());
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
  auto pinned = prepare_batch_slot(0, batch.tokens.size(), batch.loss_mask.size());
  std::copy(batch.tokens.begin(), batch.tokens.end(), pinned.tokens);
  std::copy(batch.loss_mask.begin(), batch.loss_mask.end(), pinned.mask);
  stage_batch_slot(0, batch.batch_size, batch.sequence_len, h2d_seconds);
  forward_backward_slot(0, logits != nullptr, fwd_seconds, bwd_seconds,
                        grad_scale, reset_grads);
  double loss = slot_loss(0);
  if (logits) slot_logits(0, logits);
  return loss;
}

DenseCudaPinnedBatch DenseCudaState::prepare_batch_slot(
    int slot_index, size_t token_count, size_t mask_count) {
  slot_index %= kBatchSlots;
  wait_batch_slot(slot_index);
  ensure_slot_buffers(slot_index, token_count, mask_count);
  return {slots_[slot_index].host_tokens, slots_[slot_index].host_mask};
}

void DenseCudaState::stage_batch_slot(int slot_index, int batch_size,
                                      int seq_len, double* h2d_seconds) {
  auto& slot = slots_[slot_index % kBatchSlots];
  slot.batch_size = batch_size;
  slot.seq_len = seq_len;
  slot.supervised = 0;
  for (int b = 0; b < batch_size; ++b) {
    int base = b * seq_len;
    for (int pos = 0; pos + 1 < seq_len; ++pos) {
      if (slot.host_mask[base + pos + 1] != 0) ++slot.supervised;
    }
  }
  slot.capture_logits = false;
  {
    EventSpan timer(ctx_->copy_stream(), "dense H2D event");
    size_t items = static_cast<size_t>(batch_size) * seq_len;
    require_cuda(cudaMemcpyAsync(slot.device_tokens, slot.host_tokens,
                                 items * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, ctx_->copy_stream()),
                 "tokens H2D");
    require_cuda(cudaMemcpyAsync(slot.device_mask, slot.host_mask, items,
                                 cudaMemcpyHostToDevice, ctx_->copy_stream()),
                 "mask H2D");
    require_cuda(cudaEventRecord(slot.h2d_done, ctx_->copy_stream()),
                 "batch h2d done");
    if (h2d_seconds) *h2d_seconds += timer.seconds();
  }
  slot.used = true;
}

void DenseCudaState::forward_backward_slot(int slot_index, bool capture_logits,
                                           double* fwd_seconds,
                                           double* bwd_seconds,
                                           float grad_scale,
                                           bool reset_grads) {
  auto& slot = slots_[slot_index % kBatchSlots];
  int rows = slot.batch_size * slot.seq_len;
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  ensure_step_buffers(rows);
  require_cuda(cudaStreamWaitEvent(ctx_->stream(), slot.h2d_done, 0),
               "wait batch h2d");
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
    dense_launch_gather(slot.device_tokens, emb_shadow_.data(),
                        step_hidden_.data(), slot.batch_size, slot.seq_len,
                        v, h, ctx_->stream());
    gemm(step_hidden_, step_out_, rows);
    dense_launch_loss_grad(static_cast<float*>(step_out_.data()),
                           slot.device_tokens, slot.device_mask,
                           static_cast<float*>(step_grad_logits_.data()),
                           static_cast<float*>(step_loss_.data()),
                           slot.batch_size, slot.seq_len, v,
                           slot.supervised, grad_scale,
                           ctx_->stream());
    if (fwd_seconds) *fwd_seconds += timer.seconds();
  }
  {
    EventSpan timer(ctx_->stream(), "dense backward event");
    dense_launch_bf16_to_f32(step_hidden_.data(),
                             static_cast<float*>(step_hidden_f32_.data()),
                             rows * h, ctx_->stream());
    dense_launch_bf16_to_f32(head_shadow_.data(),
                             static_cast<float*>(step_head_f32_.data()),
                             v * h, ctx_->stream());
    gemm_head_grad(step_grad_logits_, step_hidden_f32_, rows);
    gemm_d_hidden(step_grad_logits_, step_d_hidden_, rows);
    dense_launch_scatter_emb_grad(slot.device_tokens,
                                  static_cast<float*>(step_d_hidden_.data()),
                                  static_cast<float*>(grad_emb_.data()),
                                  slot.batch_size, slot.seq_len, v, h,
                                  ctx_->stream());
    if (bwd_seconds) *bwd_seconds += timer.seconds();
  }
  slot.capture_logits = capture_logits;
  require_cuda(cudaMemcpyAsync(slot.host_loss, step_loss_.data(), sizeof(float),
                               cudaMemcpyDeviceToHost, ctx_->stream()),
               "loss D2H");
  if (capture_logits) {
    size_t row = static_cast<size_t>(rows - 2) * static_cast<size_t>(v);
    auto* source = static_cast<float*>(step_out_.data()) + row;
    require_cuda(cudaMemcpyAsync(slot.host_logits, source, v * sizeof(float),
                                 cudaMemcpyDeviceToHost, ctx_->stream()),
                 "logits row D2H");
  }
  require_cuda(cudaEventRecord(slot.compute_done, ctx_->stream()),
               "batch compute done");
}

void DenseCudaState::wait_batch_slot(int slot_index) {
  auto& slot = slots_[slot_index % kBatchSlots];
  if (slot.used) require_cuda(cudaEventSynchronize(slot.compute_done),
                              "wait batch slot");
}

double DenseCudaState::slot_loss(int slot_index) {
  wait_batch_slot(slot_index);
  return slots_[slot_index % kBatchSlots].host_loss[0];
}

void DenseCudaState::slot_logits(int slot_index, std::vector<float>* logits) {
  auto& slot = slots_[slot_index % kBatchSlots];
  wait_batch_slot(slot_index);
  logits->assign(slot.host_logits, slot.host_logits + cfg_.vocab_size);
}

}  // namespace lkjai
