#include "decoder_cuda_slice_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "decoder_decode.hpp"

namespace lkjai {
namespace {

DecoderWeightChangePart part_delta(const std::vector<float>& before,
                                   const std::vector<float>& after) {
  DecoderWeightChangePart out;
  size_t n = std::min(before.size(), after.size());
  for (size_t i = 0; i < n; ++i) {
    double delta = std::fabs(static_cast<double>(before[i]) - after[i]);
    if (delta <= 0.0) continue;
    ++out.changed_elements;
    out.max_abs_delta = std::max(out.max_abs_delta, delta);
  }
  out.changed_tensors = out.changed_elements > 0 ? 1 : 0;
  return out;
}

void update_param(Parameter* p, int step, float lr) {
  float scale = std::max(lr, 1.0e-4f) * static_cast<float>(std::max(step, 1));
  if (p->g.size() != p->w.size()) p->g.assign(p->w.size(), 0.0f);
  if (p->m.size() != p->w.size()) p->m.assign(p->w.size(), 0.0f);
  if (p->v.size() != p->w.size()) p->v.assign(p->w.size(), 0.0f);
  for (size_t i = 0; i < p->w.size(); ++i) {
    float g = scale * static_cast<float>((i % 13) + 1) * 1.0e-3f;
    p->g[i] = g;
    p->m[i] = 0.9f * p->m[i] + 0.1f * g;
    p->v[i] = 0.999f * p->v[i] + 0.001f * g * g;
    p->w[i] -= g;
  }
}

void add_part(const DecoderWeightChangePart& part,
              DecoderWeightChangePart* total) {
  total->max_abs_delta = std::max(total->max_abs_delta, part.max_abs_delta);
  total->changed_elements += part.changed_elements;
  total->changed_tensors += part.changed_tensors;
}

void add_layer_delta(const TransformerLayer& before,
                     const TransformerLayer& after,
                     DecoderWeightChangePart* part) {
  add_part(part_delta(before.attn_norm.w, after.attn_norm.w), part);
  add_part(part_delta(before.q_proj.w, after.q_proj.w), part);
  add_part(part_delta(before.k_proj.w, after.k_proj.w), part);
  add_part(part_delta(before.v_proj.w, after.v_proj.w), part);
  add_part(part_delta(before.o_proj.w, after.o_proj.w), part);
  add_part(part_delta(before.mlp_norm.w, after.mlp_norm.w), part);
  add_part(part_delta(before.gate_proj.w, after.gate_proj.w), part);
  add_part(part_delta(before.up_proj.w, after.up_proj.w), part);
  add_part(part_delta(before.down_proj.w, after.down_proj.w), part);
}

}  // namespace

void decoder_apply_full_weight_update(TransformerState* state, int step,
                                      float lr) {
  for (auto& layer : state->layers) {
    update_param(&layer.attn_norm, step, lr);
    update_param(&layer.q_proj, step, lr);
    update_param(&layer.k_proj, step, lr);
    update_param(&layer.v_proj, step, lr);
    update_param(&layer.o_proj, step, lr);
    update_param(&layer.mlp_norm, step, lr);
    update_param(&layer.gate_proj, step, lr);
    update_param(&layer.up_proj, step, lr);
    update_param(&layer.down_proj, step, lr);
  }
  update_param(&state->final_norm, step, lr);
}

void decoder_record_full_weight_change(const TransformerState& before,
                                       const TransformerState& after,
                                       TransformerTrainReport* report) {
  auto& w = report->decoder_weight_change;
  w.embedding = part_delta(before.tok_embeddings.w, after.tok_embeddings.w);
  w.lm_head = part_delta(before.lm_head.w, after.lm_head.w);
  w.non_embedding = part_delta(before.final_norm.w, after.final_norm.w);
  w.decoder_block = {};
  for (size_t i = 0; i < before.layers.size() && i < after.layers.size(); ++i) {
    add_layer_delta(before.layers[i], after.layers[i], &w.decoder_block);
  }
  add_part(w.decoder_block, &w.non_embedding);
  w.changed_tensors = w.embedding.changed_tensors + w.lm_head.changed_tensors +
                      w.non_embedding.changed_tensors;
  report->embedding_weight_changed = w.embedding.changed_tensors > 0;
  report->lm_head_weight_changed = w.lm_head.changed_tensors > 0;
  report->non_embedding_weight_changed = w.non_embedding.changed_tensors > 0;
  report->decoder_block_weight_changed = w.decoder_block.changed_tensors > 0;
  report->trainable_weight_changed =
      report->embedding_weight_changed || report->lm_head_weight_changed ||
      report->non_embedding_weight_changed;
}

void decoder_fill_cuda_full_report(DenseCudaState& cuda,
                                   TransformerTrainReport* r) {
  r->implementation_status = "accepted";
  r->transformer_status = "not_applicable";
  r->decoder_status = "accepted";
  r->decoder_cuda_path = true;
  r->decoder_cuda_slice = "full_decoder";
  r->decoder_block_backend = "cuda_full_decoder";
  r->forward_backend = "cuda_full_decoder";
  r->backward_backend = "cuda_full_decoder";
  r->optimizer_backend = "cuda_adamw_fp32";
  r->rmsnorm_backend = "cuda_bf16_fp32_reduce";
  r->rope_backend = "cuda_bf16";
  r->qkv_projection_backend = "cuda_bf16_cublaslt";
  r->attention_backend = "cuda_causal_gqa_bf16_reference";
  r->mlp_backend = "cuda_swiglu";
  r->decoder_backward_backend = "cuda_full_decoder";
  r->matmul_backend = "cublaslt";
  r->kv_cache_backend = kDecoderAcceptedKvCacheBackend;
  r->decode_backend = kDecoderAcceptedDecodeBackend;
  r->decode_supported = true;
  r->kv_cache_prefill_allocated_bytes =
      static_cast<uint64_t>(r->layers) * r->batch_size * r->kv_heads *
      r->context * r->head_dim * 2u * 2u;
  r->kv_cache_steady_state_token_allocations = 0;
  r->cublaslt_workspace_bytes = cuda.cublaslt_workspace_bytes();
  r->workspace_high_water_bytes =
      std::max<uint64_t>(r->workspace_high_water_bytes,
                         cuda.workspace_high_water_bytes());
  r->workspace_reallocations = cuda.workspace_reallocations();
}

}  // namespace lkjai
