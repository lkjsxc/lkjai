#include "decoder_cuda_state.hpp"

#include <algorithm>
#include <string>

namespace lkjai {
namespace {

int element_count(const std::vector<int>& shape) {
  int out = 1;
  for (int dim : shape) out *= dim;
  return out;
}

DeviceTensor f32_tensor(cudaStream_t stream, const Parameter& p) {
  return DeviceTensor({DeviceDType::f32,
                       {static_cast<int64_t>(element_count(p.shape))}},
                      stream);
}

DeviceTensor bf16_tensor(cudaStream_t stream, const Parameter& p) {
  return DeviceTensor({DeviceDType::bf16,
                       {static_cast<int64_t>(element_count(p.shape))}},
                      stream);
}

void append_param(Parameter* p, const std::string& name,
                  const std::string& role, const std::string& tied_alias,
                  cudaStream_t stream,
                  std::vector<DecoderCudaState::RegistryTensor>* registry,
                  uint64_t* shadow_bytes) {
  DecoderCudaState::RegistryTensor t;
  t.param = p;
  t.name = name;
  t.role = role;
  t.tied_alias = tied_alias;
  t.weight = f32_tensor(stream, *p);
  t.grad = f32_tensor(stream, *p);
  t.moment_m = f32_tensor(stream, *p);
  t.moment_v = f32_tensor(stream, *p);
  t.shadow = bf16_tensor(stream, *p);
  t.weight.copy_from_host_f32(p->w, stream);
  t.grad.copy_from_host_f32(p->g, stream);
  t.moment_m.copy_from_host_f32(p->m, stream);
  t.moment_v.copy_from_host_f32(p->v, stream);
  t.shadow.copy_from_host_f32(p->w, stream);
  *shadow_bytes += t.shadow.bytes();
  registry->push_back(std::move(t));
}

template <typename Fn>
void each_param(TransformerState* state, Fn fn) {
  fn(&state->tok_embeddings);
  if (state->cfg.kind != "decoder") fn(&state->pos_embeddings);
  for (auto& layer : state->layers) {
    fn(&layer.attn_norm);
    fn(&layer.q_proj);
    fn(&layer.k_proj);
    fn(&layer.v_proj);
    fn(&layer.o_proj);
    fn(&layer.mlp_norm);
    fn(&layer.gate_proj);
    fn(&layer.up_proj);
    fn(&layer.down_proj);
  }
  fn(&state->final_norm);
  if (!state->cfg.tie_embeddings) fn(&state->lm_head);
}

}  // namespace

DecoderCudaState::DecoderCudaState(const TransformerConfig& cfg,
                                   const TransformerState& initial)
    : state_(initial),
      ctx_(),
      dense_host_(decoder_dense_state(decoder_dense_cfg(cfg), initial)),
      dense_cuda_(dense_host_.cfg, dense_host_, &ctx_) {
  build_registry();
}

TransformerState DecoderCudaState::copy_to_host() {
  copy_registry_to_host();
  return state_;
}

void DecoderCudaState::fill_report(TransformerTrainReport* report) {
  decoder_fill_full_cuda_report(dense_cuda_, registry_shadow_bytes_, report);
  report->workspace_high_water_bytes =
      std::max<uint64_t>(report->workspace_high_water_bytes,
                         dense_cuda_.workspace_high_water_bytes() +
                             registry_shadow_bytes_);
}

void DecoderCudaState::record_weight_change(const TransformerState& before,
                                            TransformerTrainReport* report) {
  auto after = copy_to_host();
  decoder_record_full_weight_change(before, after, report);
}

void DecoderCudaState::build_registry() {
  registry_.clear();
  registry_shadow_bytes_ = 0;
  append_param(&state_.tok_embeddings, "tok_embeddings", "embedding",
               state_.cfg.tie_embeddings ? "lm_head" : "", ctx_.stream(),
               &registry_, &registry_shadow_bytes_);
  for (size_t i = 0; i < state_.layers.size(); ++i) {
    auto prefix = "layers." + std::to_string(i) + ".";
    auto& layer = state_.layers[i];
    append_param(&layer.attn_norm, prefix + "attn_norm", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.q_proj, prefix + "q_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.k_proj, prefix + "k_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.v_proj, prefix + "v_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.o_proj, prefix + "o_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.mlp_norm, prefix + "mlp_norm", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.gate_proj, prefix + "gate_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.up_proj, prefix + "up_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
    append_param(&layer.down_proj, prefix + "down_proj", "decoder_block", "",
                 ctx_.stream(), &registry_, &registry_shadow_bytes_);
  }
  append_param(&state_.final_norm, "final_norm", "final_norm", "",
               ctx_.stream(), &registry_, &registry_shadow_bytes_);
  if (!state_.cfg.tie_embeddings) {
    append_param(&state_.lm_head, "lm_head", "lm_head", "", ctx_.stream(),
                 &registry_, &registry_shadow_bytes_);
  }
}

void DecoderCudaState::copy_registry_to_host() {
  for (auto& t : registry_) {
    t.param->w = t.weight.copy_to_host_f32(ctx_.stream());
    t.param->m = t.moment_m.copy_to_host_f32(ctx_.stream());
    t.param->v = t.moment_v.copy_to_host_f32(ctx_.stream());
    t.param->g = t.grad.copy_to_host_f32(ctx_.stream());
  }
}

void DecoderCudaState::sync_registry_from_host() {
  for (auto& t : registry_) {
    t.weight.copy_from_host_f32(t.param->w, ctx_.stream());
    t.grad.copy_from_host_f32(t.param->g, ctx_.stream());
    t.moment_m.copy_from_host_f32(t.param->m, ctx_.stream());
    t.moment_v.copy_from_host_f32(t.param->v, ctx_.stream());
    t.shadow.copy_from_host_f32(t.param->w, ctx_.stream());
  }
}

void DecoderCudaState::scale_and_accumulate_grads(
    const TransformerState& previous, float grad_scale, bool reset_grads) {
  size_t index = 0;
  each_param(&state_, [&](Parameter* p) {
    const Parameter* before = nullptr;
    size_t seen = 0;
    auto find = [&](const Parameter& candidate) {
      if (seen++ == index) before = &candidate;
    };
    find(previous.tok_embeddings);
    if (previous.cfg.kind != "decoder") find(previous.pos_embeddings);
    for (const auto& layer : previous.layers) {
      find(layer.attn_norm);
      find(layer.q_proj);
      find(layer.k_proj);
      find(layer.v_proj);
      find(layer.o_proj);
      find(layer.mlp_norm);
      find(layer.gate_proj);
      find(layer.up_proj);
      find(layer.down_proj);
    }
    find(previous.final_norm);
    if (!previous.cfg.tie_embeddings) find(previous.lm_head);
    for (size_t i = 0; i < p->g.size(); ++i) {
      float old = (!reset_grads && before) ? before->g[i] : 0.0f;
      p->g[i] = old + p->g[i] * grad_scale;
    }
    ++index;
  });
  if (state_.cfg.tie_embeddings) {
    for (size_t i = 0; i < state_.lm_head.g.size(); ++i) {
      float old = reset_grads ? 0.0f : previous.lm_head.g[i];
      float raw = state_.lm_head.g[i] - previous.lm_head.g[i];
      state_.lm_head.g[i] = old + raw * grad_scale;
    }
  }
}

}  // namespace lkjai
