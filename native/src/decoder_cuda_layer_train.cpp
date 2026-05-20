#include "decoder_cuda_layer_forward.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cudnn_sdpa.hpp"
#include "decoder_cuda_norm.hpp"
#include "decoder_cuda_residual.hpp"
#include "decoder_cuda_tape.hpp"
#include "transformer_train.hpp"

namespace lkjai {
namespace {

bool finite(const DeviceTensor& tensor, cudaStream_t stream) {
  auto host = tensor.copy_to_host_f32(stream);
  return std::all_of(host.begin(), host.end(),
                     [](float v) { return std::isfinite(v); });
}

void copy_bf16(const DeviceTensor& src, DeviceTensor* dst, int elements,
               cudaStream_t stream, const char* label) {
  require_cuda(cudaMemcpyAsync(dst->data(), src.data(),
                               static_cast<size_t>(elements) *
                                   sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream),
               label);
}

}  // namespace

void DecoderCudaLayerForward::run_train(
    const DeviceTensor& x, int batch, int seq, DecoderCudaLayerTape* tape,
    DecoderCudaForwardSubstrateReport* report, bool use_cudnn_attention,
    DecoderCudaRuntimeEvidence* evidence) {
  if (!tape) throw std::runtime_error("decoder CUDA training tape is null");
  DecoderCudaForwardSubstrateReport local_report;
  bool verify = report != nullptr;
  if (!report) report = &local_report;
  auto checked = [&](const DeviceTensor& tensor) {
    return verify ? finite(tensor, ctx_->stream()) : true;
  };
  int rows = batch * seq;
  int hidden_elems = rows * cfg_.hidden_size;
  void* ws = workspace_->allocate(workspace_bytes_);
  copy_bf16(x, &tape->attn_norm_input, hidden_elems, ctx_->stream(),
            "decoder train attention norm input copy");
  decoder_launch_rmsnorm_bf16(
      tape->attn_norm_input.data(), static_cast<float*>(attn_w_.data()),
      tape->attn_norm.data(), rows, cfg_.hidden_size, cfg_.rms_norm_eps,
      ctx_->stream());
  report->rmsnorm_checked = checked(tape->attn_norm);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->attn_norm.data(), wq_.data(),
                            tape->q_rope.data(), rows, cfg_.hidden_size,
                            cfg_.hidden_size, ws, workspace_bytes_);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->attn_norm.data(), wk_.data(),
                            tape->k_rope.data(), rows, cfg_.hidden_size,
                            kv_width_, ws, workspace_bytes_);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->attn_norm.data(), wv_.data(),
                            tape->v.data(), rows, cfg_.hidden_size, kv_width_,
                            ws, workspace_bytes_);
  report->qkv_projection_checked =
      checked(tape->q_rope) && checked(tape->k_rope) && checked(tape->v);
  decoder_launch_rope_bf16_at(tape->q_rope.data(), batch, seq, cfg_.heads,
                              cfg_.head_dim, 0, cfg_.rope_theta,
                              ctx_->stream());
  decoder_launch_rope_bf16_at(tape->k_rope.data(), batch, seq, cfg_.kv_heads,
                              cfg_.head_dim, 0, cfg_.rope_theta,
                              ctx_->stream());
  report->rope_checked = checked(tape->q_rope) && checked(tape->k_rope);
  if (use_cudnn_attention) {
    DecoderCudnnSdpaStats sdpa;
    decoder_cudnn_sdpa_forward_bf16_gqa(
        ctx_->cudnn(), workspace_, tape->q_rope.data(), tape->k_rope.data(),
        tape->v.data(), tape->attention_state.data(), tape->sdpa_stats.data(),
        {batch, seq, cfg_.heads, cfg_.kv_heads, cfg_.head_dim}, &sdpa);
    if (evidence) {
      evidence->cudnn_sdpa_forward_count += sdpa.executed ? 1 : 0;
      evidence->cudnn_sdpa_plan_cache_hits += sdpa.plan_cache_hit ? 1 : 0;
      evidence->cudnn_sdpa_plan_cache_misses += sdpa.plan_cache_miss ? 1 : 0;
      evidence->cudnn_sdpa_workspace_bytes =
          std::max(evidence->cudnn_sdpa_workspace_bytes, sdpa.workspace_bytes);
    }
  } else {
    decoder_launch_causal_gqa_attention_bf16(
        tape->q_rope.data(), tape->k_rope.data(), tape->v.data(),
        tape->attention_state.data(), batch, seq, cfg_.heads, cfg_.kv_heads,
        cfg_.head_dim, ctx_->stream());
    if (evidence) ++evidence->attention_reference_forward_count;
  }
  report->attention_checked = checked(tape->attention_state);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->attention_state.data(), wo_.data(),
                            tape->o_proj.data(), rows, cfg_.hidden_size,
                            cfg_.hidden_size, ws, workspace_bytes_);
  report->o_projection_checked = checked(tape->o_proj);
  decoder_launch_residual_add_bf16(
      tape->attn_norm_input.data(), tape->o_proj.data(),
      tape->attention_residual.data(), hidden_elems, ctx_->stream());
  report->attention_residual_checked = checked(tape->attention_residual);
  copy_bf16(tape->attention_residual, &tape->mlp_norm_input, hidden_elems,
            ctx_->stream(), "decoder train mlp norm input copy");
  decoder_launch_rmsnorm_bf16(
      tape->mlp_norm_input.data(), static_cast<float*>(mlp_w_.data()),
      tape->mlp_norm.data(), rows, cfg_.hidden_size, cfg_.rms_norm_eps,
      ctx_->stream());
  report->mlp_norm_checked = checked(tape->mlp_norm);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->mlp_norm.data(), wg_.data(),
                            tape->gate.data(), rows, cfg_.hidden_size,
                            cfg_.ffn_size, ws, workspace_bytes_);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->mlp_norm.data(), wu_.data(),
                            tape->up.data(), rows, cfg_.hidden_size,
                            cfg_.ffn_size, ws, workspace_bytes_);
  decoder_launch_swiglu_bf16(tape->gate.data(), tape->up.data(),
                             tape->swiglu.data(), rows * cfg_.ffn_size,
                             ctx_->stream());
  report->swiglu_checked = checked(tape->swiglu);
  decoder_cuda_project_bf16(ctx_->cublaslt(), ctx_->stream(),
                            tape->swiglu.data(), wd_.data(),
                            tape->down.data(), rows, cfg_.ffn_size,
                            cfg_.hidden_size, ws, workspace_bytes_);
  report->down_projection_checked = checked(tape->down);
  decoder_launch_residual_add_bf16(
      tape->mlp_norm_input.data(), tape->down.data(),
      tape->block_residual.data(), hidden_elems, ctx_->stream());
  report->block_residual_checked = checked(tape->block_residual);
  report->outputs_finite = report->rmsnorm_checked && report->rope_checked &&
                           report->qkv_projection_checked &&
                           report->attention_checked &&
                           report->o_projection_checked &&
                           report->attention_residual_checked &&
                           report->mlp_norm_checked && report->swiglu_checked &&
                           report->down_projection_checked &&
                           report->block_residual_checked;
}

}  // namespace lkjai
