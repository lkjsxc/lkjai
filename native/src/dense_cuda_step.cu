#include "dense_cuda_internal.hpp"

#include <cuda_runtime.h>

namespace lkjai {

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
  step_head_f32_valid_ = false;
}

double DenseCudaState::forward_backward(const PackedBatch& batch,
                                        std::vector<float>* logits,
                                        double* h2d_seconds,
                                        double* fwd_seconds,
                                        double* bwd_seconds,
                                        float grad_scale,
                                        bool reset_grads) {
  auto pinned = prepare_batch_slot(0, batch.tokens.size(), batch.loss_mask.size());
  std::copy(batch.tokens.begin(), batch.tokens.end(), pinned.tokens);
  std::copy(batch.loss_mask.begin(), batch.loss_mask.end(), pinned.mask);
  stage_batch_slot(0, batch.batch_size, batch.sequence_len, h2d_seconds);
  forward_backward_slot(0, logits != nullptr, fwd_seconds, bwd_seconds,
                        grad_scale, reset_grads);
  double loss = slot_loss(0);
  take_deferred_timings(h2d_seconds, fwd_seconds, bwd_seconds);
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
  (void)h2d_seconds;
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
  slot.timings_accounted = false;
  size_t items = static_cast<size_t>(batch_size) * seq_len;
  require_cuda(cudaEventRecord(slot.h2d_start, ctx_->copy_stream()),
               "batch h2d start");
  require_cuda(cudaMemcpyAsync(slot.device_tokens, slot.host_tokens,
                               items * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, ctx_->copy_stream()),
               "tokens H2D");
  require_cuda(cudaMemcpyAsync(slot.device_mask, slot.host_mask, items,
                               cudaMemcpyHostToDevice, ctx_->copy_stream()),
               "mask H2D");
  require_cuda(cudaEventRecord(slot.h2d_done, ctx_->copy_stream()),
               "batch h2d done");
  slot.used = true;
}

void DenseCudaState::forward_backward_slot(int slot_index, bool capture_logits,
                                           double* fwd_seconds,
                                           double* bwd_seconds,
                                           float grad_scale,
                                           bool reset_grads) {
  (void)fwd_seconds;
  (void)bwd_seconds;
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
  require_cuda(cudaEventRecord(slot.forward_start, ctx_->stream()),
               "dense forward start");
  dense_launch_gather(slot.device_tokens, emb_shadow_.data(),
                      step_hidden_.data(), slot.batch_size, slot.seq_len,
                      v, h, ctx_->stream());
  gemm(step_hidden_, step_out_, rows);
  dense_launch_loss_grad(static_cast<float*>(step_out_.data()),
                         slot.device_tokens, slot.device_mask,
                         static_cast<float*>(step_grad_logits_.data()),
                         static_cast<float*>(step_loss_.data()),
                         slot.batch_size, slot.seq_len, v, slot.supervised,
                         grad_scale, ctx_->stream());
  require_cuda(cudaEventRecord(slot.forward_done, ctx_->stream()),
               "dense forward done");
  require_cuda(cudaEventRecord(slot.backward_start, ctx_->stream()),
               "dense backward start");
  dense_launch_bf16_to_f32(step_hidden_.data(),
                           static_cast<float*>(step_hidden_f32_.data()),
                           rows * h, ctx_->stream());
  if (!step_head_f32_valid_) {
    dense_launch_bf16_to_f32(head_shadow_.data(),
                             static_cast<float*>(step_head_f32_.data()),
                             v * h, ctx_->stream());
    step_head_f32_valid_ = true;
    ++head_f32_cache_refreshes_;
  }
  gemm_head_grad(step_grad_logits_, step_hidden_f32_, rows);
  gemm_d_hidden(step_grad_logits_, step_d_hidden_, rows);
  dense_launch_scatter_emb_grad(slot.device_tokens,
                                static_cast<float*>(step_d_hidden_.data()),
                                static_cast<float*>(grad_emb_.data()),
                                slot.batch_size, slot.seq_len, v, h,
                                ctx_->stream());
  require_cuda(cudaEventRecord(slot.backward_done, ctx_->stream()),
               "dense backward done");
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
  if (!slot.used) return;
  require_cuda(cudaEventSynchronize(slot.compute_done), "wait batch slot");
  if (slot.timings_accounted) return;
  float h2d = 0.0f, fwd = 0.0f, bwd = 0.0f;
  require_cuda(cudaEventElapsedTime(&h2d, slot.h2d_start, slot.h2d_done),
               "elapsed h2d");
  require_cuda(cudaEventElapsedTime(&fwd, slot.forward_start, slot.forward_done),
               "elapsed forward");
  require_cuda(cudaEventElapsedTime(&bwd, slot.backward_start, slot.backward_done),
               "elapsed backward");
  pending_h2d_seconds_ += static_cast<double>(h2d) / 1000.0;
  pending_forward_seconds_ += static_cast<double>(fwd) / 1000.0;
  pending_backward_seconds_ += static_cast<double>(bwd) / 1000.0;
  slot.timings_accounted = true;
}

void DenseCudaState::take_deferred_timings(double* h2d, double* forward,
                                           double* backward) {
  if (h2d) *h2d += pending_h2d_seconds_;
  if (forward) *forward += pending_forward_seconds_;
  if (backward) *backward += pending_backward_seconds_;
  pending_h2d_seconds_ = 0.0;
  pending_forward_seconds_ = 0.0;
  pending_backward_seconds_ = 0.0;
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
