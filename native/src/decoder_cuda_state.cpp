#include "decoder_cuda_state.hpp"

#include <algorithm>

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

void append_param(Parameter* p, cudaStream_t stream,
                  std::vector<DecoderCudaState::RegistryTensor>* registry,
                  uint64_t* shadow_bytes) {
  DecoderCudaState::RegistryTensor t;
  t.param = p;
  t.weight = f32_tensor(stream, *p);
  t.moment_m = f32_tensor(stream, *p);
  t.moment_v = f32_tensor(stream, *p);
  t.shadow = bf16_tensor(stream, *p);
  t.weight.copy_from_host_f32(p->w, stream);
  t.moment_m.copy_from_host_f32(p->m, stream);
  t.moment_v.copy_from_host_f32(p->v, stream);
  t.shadow.copy_from_host_f32(p->w, stream);
  *shadow_bytes += t.shadow.bytes();
  registry->push_back(std::move(t));
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
  auto dense = dense_cuda_.copy_to_host();
  decoder_copy_dense_back(dense, &state_);
  copy_registry_to_host();
  return state_;
}

void DecoderCudaState::fill_report(TransformerTrainReport* report) {
  decoder_fill_cuda_slice_report(dense_cuda_, report);
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
  for (auto& layer : state_.layers) {
    append_param(&layer.attn_norm, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.q_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.k_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.v_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.o_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.mlp_norm, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.gate_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.up_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
    append_param(&layer.down_proj, ctx_.stream(), &registry_,
                 &registry_shadow_bytes_);
  }
  append_param(&state_.final_norm, ctx_.stream(), &registry_,
               &registry_shadow_bytes_);
}

void DecoderCudaState::copy_registry_to_host() {
  for (auto& t : registry_) {
    t.param->w = t.weight.copy_to_host_f32(ctx_.stream());
    t.param->m = t.moment_m.copy_to_host_f32(ctx_.stream());
    t.param->v = t.moment_v.copy_to_host_f32(ctx_.stream());
  }
}

}  // namespace lkjai
