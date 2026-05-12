#include "decoder_cuda_layer_forward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_norm.hpp"
#include "decoder_cuda_residual.hpp"

namespace lkjai {
namespace {

DeviceTensor bf16(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
}

bool finite(const DeviceTensor& tensor, cudaStream_t stream) {
  auto host = tensor.copy_to_host_f32(stream);
  return std::all_of(host.begin(), host.end(),
                     [](float v) { return std::isfinite(v); });
}

std::vector<float> projection_weight(const Parameter& p, int in, int out) {
  std::vector<float> t(static_cast<size_t>(in) * out);
  for (int i = 0; i < in; ++i) {
    for (int o = 0; o < out; ++o) {
      t[static_cast<size_t>(o) * in + i] =
          p.w[static_cast<size_t>(i) * out + o];
    }
  }
  return t;
}

}  // namespace

DecoderCudaLayerForward::DecoderCudaLayerForward(
    const TransformerConfig& cfg, const TransformerLayer& layer,
    CudaExecutionContext* ctx, DeviceWorkspace* workspace,
    size_t workspace_bytes)
    : cfg_(cfg),
      ctx_(ctx),
      workspace_(workspace),
      workspace_bytes_(workspace_bytes),
      kv_width_(cfg.kv_heads * cfg.head_dim),
      attn_w_({DeviceDType::f32, {cfg.hidden_size}}, ctx->stream()),
      mlp_w_({DeviceDType::f32, {cfg.hidden_size}}, ctx->stream()),
      wq_(bf16(ctx->stream(), cfg.hidden_size, cfg.hidden_size)),
      wk_(bf16(ctx->stream(), kv_width_, cfg.hidden_size)),
      wv_(bf16(ctx->stream(), kv_width_, cfg.hidden_size)),
      wo_(bf16(ctx->stream(), cfg.hidden_size, cfg.hidden_size)),
      wg_(bf16(ctx->stream(), cfg.ffn_size, cfg.hidden_size)),
      wu_(bf16(ctx->stream(), cfg.ffn_size, cfg.hidden_size)),
      wd_(bf16(ctx->stream(), cfg.hidden_size, cfg.ffn_size)) {
  attn_w_.copy_from_host_f32(layer.attn_norm.w, ctx_->stream());
  mlp_w_.copy_from_host_f32(layer.mlp_norm.w, ctx_->stream());
  upload_projection(&wq_, layer.q_proj, cfg_.hidden_size, cfg_.hidden_size);
  upload_projection(&wk_, layer.k_proj, cfg_.hidden_size, kv_width_);
  upload_projection(&wv_, layer.v_proj, cfg_.hidden_size, kv_width_);
  upload_projection(&wo_, layer.o_proj, cfg_.hidden_size, cfg_.hidden_size);
  upload_projection(&wg_, layer.gate_proj, cfg_.hidden_size, cfg_.ffn_size);
  upload_projection(&wu_, layer.up_proj, cfg_.hidden_size, cfg_.ffn_size);
  upload_projection(&wd_, layer.down_proj, cfg_.ffn_size, cfg_.hidden_size);
}

void DecoderCudaLayerForward::upload_projection(DeviceTensor* dst,
                                                const Parameter& src, int in,
                                                int out) {
  dst->copy_from_host_f32(projection_weight(src, in, out), ctx_->stream());
}

void DecoderCudaLayerForward::allocate_scratch(int rows) {
  if (scratch_rows_ == rows) return;
  auto stream = ctx_->stream();
  norm_ = bf16(stream, rows, cfg_.hidden_size);
  q_ = bf16(stream, rows, cfg_.hidden_size);
  k_ = bf16(stream, rows, kv_width_);
  v_ = bf16(stream, rows, kv_width_);
  attn_ = bf16(stream, rows, cfg_.hidden_size);
  o_ = bf16(stream, rows, cfg_.hidden_size);
  attn_resid_ = bf16(stream, rows, cfg_.hidden_size);
  mlp_norm_ = bf16(stream, rows, cfg_.hidden_size);
  gate_ = bf16(stream, rows, cfg_.ffn_size);
  up_ = bf16(stream, rows, cfg_.ffn_size);
  swiglu_ = bf16(stream, rows, cfg_.ffn_size);
  down_ = bf16(stream, rows, cfg_.hidden_size);
  scratch_rows_ = rows;
}

void DecoderCudaLayerForward::run(
    const DeviceTensor& x, int batch, int seq, DeviceTensor* out,
    DecoderCudaForwardSubstrateReport* report) {
  run(x, batch, seq, out, report, nullptr);
}

void DecoderCudaLayerForward::run(
    const DeviceTensor& x, int batch, int seq, DeviceTensor* out,
    DecoderCudaForwardSubstrateReport* report,
    const DecoderCudaLayerCacheView* cache) {
  int rows = batch * seq;
  allocate_scratch(rows);
  *out = bf16(ctx_->stream(), rows, cfg_.hidden_size);
  void* ws = workspace_->allocate(workspace_bytes_);
  decoder_launch_rmsnorm_bf16(x.data(), static_cast<float*>(attn_w_.data()),
                              norm_.data(), rows, cfg_.hidden_size,
                              cfg_.rms_norm_eps, ctx_->stream());
  report->rmsnorm_checked = finite(norm_, ctx_->stream());
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), norm_.data(),
                            wq_.data(), q_.data(), rows, cfg_.hidden_size,
                            cfg_.hidden_size, ws, workspace_bytes_);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), norm_.data(),
                            wk_.data(), k_.data(), rows, cfg_.hidden_size,
                            kv_width_, ws, workspace_bytes_);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), norm_.data(),
                            wv_.data(), v_.data(), rows, cfg_.hidden_size,
                            kv_width_, ws, workspace_bytes_);
  report->qkv_projection_checked = finite(q_, ctx_->stream()) &&
                                   finite(k_, ctx_->stream()) &&
                                   finite(v_, ctx_->stream());
  int pos = cache ? cache->start_position : 0;
  decoder_launch_rope_bf16_at(q_.data(), batch, seq, cfg_.heads, cfg_.head_dim,
                              pos, cfg_.rope_theta, ctx_->stream());
  decoder_launch_rope_bf16_at(k_.data(), batch, seq, cfg_.kv_heads,
                              cfg_.head_dim, pos, cfg_.rope_theta,
                              ctx_->stream());
  report->rope_checked = finite(q_, ctx_->stream()) && finite(k_, ctx_->stream());
  if (cache && cache->cache) {
    std::string error;
    if (!decoder_kv_cache_append_device_layer(cache->cache, cache->layer, pos,
                                              k_.data(), v_.data(), batch,
                                              seq, ctx_->stream(), &error)) {
      throw std::runtime_error(error);
    }
  }
  if (cache && cache->cached_attention) {
    decoder_launch_cached_gqa_attention_bf16(
        q_.data(), cache->cache->key_device, cache->cache->value_device,
        attn_.data(), cache->layer, 0, pos, cache->cache->layout.cfg.batch,
        cache->cache->layout.cfg.context, batch, cfg_.heads, cfg_.kv_heads,
        cfg_.head_dim, ctx_->stream());
  } else {
    decoder_launch_causal_gqa_attention_bf16(
        q_.data(), k_.data(), v_.data(), attn_.data(), batch, seq, cfg_.heads,
        cfg_.kv_heads, cfg_.head_dim, ctx_->stream());
  }
  report->attention_checked = finite(attn_, ctx_->stream());
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), attn_.data(),
                            wo_.data(), o_.data(), rows, cfg_.hidden_size,
                            cfg_.hidden_size, ws, workspace_bytes_);
  report->o_projection_checked = finite(o_, ctx_->stream());
  decoder_launch_residual_add_bf16(x.data(), o_.data(), attn_resid_.data(),
                                   rows * cfg_.hidden_size, ctx_->stream());
  report->attention_residual_checked = finite(attn_resid_, ctx_->stream());
  decoder_launch_rmsnorm_bf16(attn_resid_.data(),
                              static_cast<float*>(mlp_w_.data()),
                              mlp_norm_.data(), rows, cfg_.hidden_size,
                              cfg_.rms_norm_eps, ctx_->stream());
  report->mlp_norm_checked = finite(mlp_norm_, ctx_->stream());
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), mlp_norm_.data(),
                            wg_.data(), gate_.data(), rows, cfg_.hidden_size,
                            cfg_.ffn_size, ws, workspace_bytes_);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), mlp_norm_.data(),
                            wu_.data(), up_.data(), rows, cfg_.hidden_size,
                            cfg_.ffn_size, ws, workspace_bytes_);
  decoder_launch_swiglu_bf16(gate_.data(), up_.data(), swiglu_.data(),
                             rows * cfg_.ffn_size, ctx_->stream());
  report->swiglu_checked = finite(swiglu_, ctx_->stream());
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(), swiglu_.data(),
                            wd_.data(), down_.data(), rows, cfg_.ffn_size,
                            cfg_.hidden_size, ws, workspace_bytes_);
  report->down_projection_checked = finite(down_, ctx_->stream());
  decoder_launch_residual_add_bf16(attn_resid_.data(), down_.data(),
                                   out->data(), rows * cfg_.hidden_size,
                                   ctx_->stream());
  report->block_residual_checked = finite(*out, ctx_->stream());
  report->outputs_finite = report->rmsnorm_checked && report->rope_checked &&
                           report->qkv_projection_checked &&
                           report->attention_checked &&
                           report->o_projection_checked &&
                           report->attention_residual_checked &&
                           report->mlp_norm_checked && report->swiglu_checked &&
                           report->down_projection_checked &&
                           report->block_residual_checked;
}

uint64_t DecoderCudaLayerForward::workspace_high_water_bytes() const {
  return workspace_->high_water_bytes();
}

}  // namespace lkjai
