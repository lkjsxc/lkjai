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
  if (host_tokens_) cudaFreeHost(host_tokens_);
  if (host_mask_) cudaFreeHost(host_mask_);
  if (device_tokens_) cudaFree(device_tokens_);
  if (device_mask_) cudaFree(device_mask_);
}

void DenseCudaState::ensure_batch_buffers(size_t token_count,
                                          size_t mask_count) {
  if (token_count > host_token_capacity_) {
    if (host_tokens_) cudaFreeHost(host_tokens_);
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&host_tokens_),
                                token_count * sizeof(uint16_t)),
                 "cudaMallocHost tokens");
    host_token_capacity_ = token_count;
  }
  if (mask_count > host_mask_capacity_) {
    if (host_mask_) cudaFreeHost(host_mask_);
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&host_mask_),
                                mask_count),
                 "cudaMallocHost mask");
    host_mask_capacity_ = mask_count;
  }
  if (token_count > device_token_capacity_) {
    if (device_tokens_) cudaFree(device_tokens_);
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&device_tokens_),
                            token_count * sizeof(uint16_t)),
                 "cudaMalloc device tokens");
    device_token_capacity_ = token_count;
  }
  if (mask_count > device_mask_capacity_) {
    if (device_mask_) cudaFree(device_mask_);
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&device_mask_),
                            mask_count),
                 "cudaMalloc device mask");
    device_mask_capacity_ = mask_count;
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
