#include "dense_cuda_internal.hpp"

#include <algorithm>

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace lkjai {

DenseCudaState::DenseCudaState(const DenseConfig& cfg,
                               const DenseTrainState& host,
                               CudaExecutionContext* ctx)
    : cfg_(cfg),
      ctx_(ctx),
      workspace_(ctx->stream()),
      emb_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
           ctx->stream()),
      head_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
            ctx->stream()),
      emb_shadow_({DeviceDType::bf16, {cfg.vocab_size, cfg.hidden_size}},
                  ctx->stream()),
      head_shadow_({DeviceDType::bf16, {cfg.vocab_size, cfg.hidden_size}},
                   ctx->stream()),
      grad_emb_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
                ctx->stream()),
      grad_head_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
                 ctx->stream()),
      m_emb_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
             ctx->stream()),
      v_emb_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
             ctx->stream()),
      m_head_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
              ctx->stream()),
      v_head_({DeviceDType::f32, {cfg.vocab_size, cfg.hidden_size}},
              ctx->stream()) {
  emb_.copy_from_host_f32(host.emb, ctx->stream());
  head_.copy_from_host_f32(host.head, ctx->stream());
  emb_shadow_.copy_from_host_f32(host.emb, ctx->stream());
  head_shadow_.copy_from_host_f32(host.head, ctx->stream());
  m_emb_.copy_from_host_f32(host.m_emb, ctx->stream());
  v_emb_.copy_from_host_f32(host.v_emb, ctx->stream());
  m_head_.copy_from_host_f32(host.m_head, ctx->stream());
  v_head_.copy_from_host_f32(host.v_head, ctx->stream());
  zero_gradients();
}

DenseCudaState::~DenseCudaState() {
  destroy_dense_matmul_plan(logits_plan_);
  destroy_dense_matmul_plan(head_grad_plan_);
  destroy_dense_matmul_plan(hidden_grad_plan_);
  for (auto& slot : slots_) {
    if (slot.compute_done) cudaEventSynchronize(slot.compute_done);
    if (slot.host_tokens) cudaFreeHost(slot.host_tokens);
    if (slot.host_mask) cudaFreeHost(slot.host_mask);
    if (slot.host_loss) cudaFreeHost(slot.host_loss);
    if (slot.host_logits) cudaFreeHost(slot.host_logits);
    if (slot.device_tokens) cudaFree(slot.device_tokens);
    if (slot.device_mask) cudaFree(slot.device_mask);
    if (slot.h2d_done) cudaEventDestroy(slot.h2d_done);
    if (slot.compute_done) cudaEventDestroy(slot.compute_done);
  }
}

void DenseCudaState::ensure_slot_buffers(int slot_index, size_t token_count,
                                         size_t mask_count) {
  auto& slot = slots_[slot_index % kBatchSlots];
  if (!slot.h2d_done) require_cuda(cudaEventCreateWithFlags(
      &slot.h2d_done, cudaEventDisableTiming), "batch h2d event");
  if (!slot.compute_done) require_cuda(cudaEventCreateWithFlags(
      &slot.compute_done, cudaEventDisableTiming), "batch compute event");
  if (!slot.host_loss) require_cuda(cudaMallocHost(
      reinterpret_cast<void**>(&slot.host_loss), sizeof(float)), "pinned loss");
  if (token_count > slot.token_capacity) {
    if (slot.host_tokens) cudaFreeHost(slot.host_tokens);
    if (slot.device_tokens) cudaFree(slot.device_tokens);
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.host_tokens),
                                token_count * sizeof(uint16_t)), "host tokens");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&slot.device_tokens),
                            token_count * sizeof(uint16_t)), "device tokens");
    slot.token_capacity = token_count;
  }
  if (mask_count > slot.mask_capacity) {
    if (slot.host_mask) cudaFreeHost(slot.host_mask);
    if (slot.device_mask) cudaFree(slot.device_mask);
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.host_mask),
                                mask_count), "host mask");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&slot.device_mask),
                            mask_count), "device mask");
    slot.mask_capacity = mask_count;
  }
  size_t logits = static_cast<size_t>(cfg_.vocab_size);
  if (logits > slot.logits_capacity) {
    if (slot.host_logits) cudaFreeHost(slot.host_logits);
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.host_logits),
                                logits * sizeof(float)), "pinned logits");
    slot.logits_capacity = logits;
  }
}

void DenseCudaState::zero_gradients() {
  for (auto* t : {&grad_emb_, &grad_head_}) {
    require_cuda(cudaMemsetAsync(t->data(), 0, t->bytes(), ctx_->stream()),
                 "zero dense tensor");
  }
}

void DenseCudaState::adamw(float lr, int step) {
  int n = cfg_.vocab_size * cfg_.hidden_size;
  dense_launch_adamw(static_cast<float*>(emb_.data()),
                     static_cast<float*>(m_emb_.data()),
                     static_cast<float*>(v_emb_.data()),
                     static_cast<float*>(grad_emb_.data()), emb_shadow_.data(),
                     n, lr, step, ctx_->stream());
  dense_launch_adamw(static_cast<float*>(head_.data()),
                     static_cast<float*>(m_head_.data()),
                     static_cast<float*>(v_head_.data()),
                     static_cast<float*>(grad_head_.data()), head_shadow_.data(),
                     n, lr, step, ctx_->stream());
}

DenseTrainState DenseCudaState::copy_to_host() {
  DenseTrainState host;
  host.cfg = cfg_;
  host.emb = emb_.copy_to_host_f32(ctx_->stream());
  host.head = head_.copy_to_host_f32(ctx_->stream());
  host.grad_emb = grad_emb_.copy_to_host_f32(ctx_->stream());
  host.grad_head = grad_head_.copy_to_host_f32(ctx_->stream());
  host.m_emb = m_emb_.copy_to_host_f32(ctx_->stream());
  host.v_emb = v_emb_.copy_to_host_f32(ctx_->stream());
  host.m_head = m_head_.copy_to_host_f32(ctx_->stream());
  host.v_head = v_head_.copy_to_host_f32(ctx_->stream());
  return host;
}

}  // namespace lkjai
