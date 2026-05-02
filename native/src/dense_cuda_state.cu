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
  require_cuda(cudaStreamSynchronize(ctx_->stream()), "zero dense sync");
}

double DenseCudaState::forward_backward(const PackedBatch& batch,
                                        std::vector<float>* logits,
                                        double* h2d_seconds,
                                        double* fwd_seconds,
                                        double* bwd_seconds,
                                        float grad_scale,
                                        bool reset_grads) {
  int rows = batch.batch_size * (batch.sequence_len - 1);
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  ensure_batch_buffers(batch.tokens.size(), batch.loss_mask.size());
  DeviceTensor hidden({DeviceDType::bf16, {rows, h}}, ctx_->stream());
  DeviceTensor out({DeviceDType::f32, {rows, v}}, ctx_->stream());
  DeviceTensor grad_logits({DeviceDType::f32, {rows, v}}, ctx_->stream());
  DeviceTensor d_hidden({DeviceDType::f32, {rows, h}}, ctx_->stream());
  DeviceTensor loss({DeviceDType::f32, {1}}, ctx_->stream());
  auto phase = std::chrono::steady_clock::now();
  std::copy(batch.tokens.begin(), batch.tokens.end(), host_tokens_);
  std::copy(batch.loss_mask.begin(), batch.loss_mask.end(), host_mask_);
  require_cuda(cudaMemcpyAsync(device_tokens_, host_tokens_,
                               batch.tokens.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, ctx_->stream()),
               "tokens H2D");
  require_cuda(cudaMemcpyAsync(device_mask_, host_mask_, batch.loss_mask.size(),
                               cudaMemcpyHostToDevice, ctx_->stream()),
               "mask H2D");
  require_cuda(cudaStreamSynchronize(ctx_->stream()), "batch H2D sync");
  if (h2d_seconds) *h2d_seconds += dense_seconds_since(phase);
  require_cuda(cudaMemsetAsync(loss.data(), 0, sizeof(float), ctx_->stream()),
               "loss memset");
  if (reset_grads) {
    require_cuda(cudaMemsetAsync(grad_emb_.data(), 0, grad_emb_.bytes(),
                                 ctx_->stream()),
                 "grad emb memset");
    require_cuda(cudaMemsetAsync(grad_head_.data(), 0, grad_head_.bytes(),
                                 ctx_->stream()),
                 "grad head memset");
  }
  phase = std::chrono::steady_clock::now();
  dense_launch_gather(device_tokens_, emb_shadow_.data(),
                      hidden.data(), batch.batch_size, batch.sequence_len, v, h,
                      ctx_->stream());
  gemm(hidden, out, rows);
  dense_launch_loss_grad(static_cast<float*>(out.data()),
                         device_tokens_, device_mask_,
                         static_cast<float*>(grad_logits.data()),
                         static_cast<float*>(loss.data()), batch.batch_size,
                         batch.sequence_len, v, dense_supervised_count(batch),
                         grad_scale, ctx_->stream());
  require_cuda(cudaStreamSynchronize(ctx_->stream()), "forward sync");
  if (fwd_seconds) *fwd_seconds += dense_seconds_since(phase);
  phase = std::chrono::steady_clock::now();
  gemm_head_grad(grad_logits, hidden, rows);
  gemm_d_hidden(grad_logits, d_hidden, rows);
  dense_launch_emb_scatter(static_cast<float*>(d_hidden.data()), device_tokens_,
                           static_cast<float*>(grad_emb_.data()),
                           batch.batch_size, batch.sequence_len, v, h,
                           ctx_->stream());
  require_cuda(cudaStreamSynchronize(ctx_->stream()), "backward sync");
  if (bwd_seconds) *bwd_seconds += dense_seconds_since(phase);
  auto loss_host = loss.copy_to_host_f32(ctx_->stream());
  if (logits) {
    auto all = out.copy_to_host_f32(ctx_->stream());
    logits->assign(all.end() - v, all.end());
  }
  return loss_host.empty() ? 0.0 : loss_host[0];
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
  require_cuda(cudaStreamSynchronize(ctx_->stream()), "adamw sync");
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
