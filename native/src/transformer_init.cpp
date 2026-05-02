#include "transformer_state.hpp"

#include <cstdint>
#include <random>

namespace lkjai {
namespace {

uint64_t elements(const std::vector<int>& shape) {
  uint64_t total = 1;
  for (int dim : shape) total *= static_cast<uint64_t>(dim);
  return total;
}

void init_param(Parameter* p, const std::string& name,
                const std::vector<int>& shape, std::mt19937* rng,
                float scale) {
  p->name = name;
  p->shape = shape;
  auto count = static_cast<size_t>(elements(shape));
  p->w.resize(count);
  p->g.assign(count, 0.0f);
  p->m.assign(count, 0.0f);
  p->v.assign(count, 0.0f);
  std::normal_distribution<float> dist(0.0f, scale);
  for (float& value : p->w) value = dist(*rng);
}

void init_norm(Parameter* p, const std::string& name, int hidden) {
  p->name = name;
  p->shape = {hidden};
  p->w.assign(static_cast<size_t>(hidden), 1.0f);
  p->g.assign(static_cast<size_t>(hidden), 0.0f);
  p->m.assign(static_cast<size_t>(hidden), 0.0f);
  p->v.assign(static_cast<size_t>(hidden), 0.0f);
}

}  // namespace

void init_transformer_state(const TransformerConfig& cfg,
                            TransformerState* state) {
  state->cfg = cfg;
  std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
  init_param(&state->tok_embeddings, "tok_embeddings",
             {cfg.vocab_size, cfg.hidden_size}, &rng, 0.02f);
  init_param(&state->pos_embeddings, "pos_embeddings",
             {cfg.context, cfg.hidden_size}, &rng, 0.02f);
  state->layers.resize(static_cast<size_t>(cfg.layers));
  int kv = cfg.kv_heads * cfg.head_dim;
  for (int i = 0; i < cfg.layers; ++i) {
    auto p = "layers." + std::to_string(i) + ".";
    auto& l = state->layers[static_cast<size_t>(i)];
    init_norm(&l.attn_norm, p + "attn_norm", cfg.hidden_size);
    init_param(&l.q_proj, p + "attn.q_proj",
               {cfg.hidden_size, cfg.hidden_size}, &rng, 0.02f);
    init_param(&l.k_proj, p + "attn.k_proj", {cfg.hidden_size, kv}, &rng,
               0.02f);
    init_param(&l.v_proj, p + "attn.v_proj", {cfg.hidden_size, kv}, &rng,
               0.02f);
    init_param(&l.o_proj, p + "attn.o_proj",
               {cfg.hidden_size, cfg.hidden_size}, &rng, 0.02f);
    init_norm(&l.mlp_norm, p + "mlp_norm", cfg.hidden_size);
    init_param(&l.gate_proj, p + "mlp.gate_proj",
               {cfg.hidden_size, cfg.ffn_size}, &rng, 0.02f);
    init_param(&l.up_proj, p + "mlp.up_proj",
               {cfg.hidden_size, cfg.ffn_size}, &rng, 0.02f);
    init_param(&l.down_proj, p + "mlp.down_proj",
               {cfg.ffn_size, cfg.hidden_size}, &rng, 0.02f);
  }
  init_norm(&state->final_norm, "final_norm", cfg.hidden_size);
  init_param(&state->lm_head, "lm_head", {cfg.vocab_size, cfg.hidden_size},
             &rng, 0.02f);
  if (cfg.tie_embeddings) state->lm_head.w = state->tok_embeddings.w;
}

}  // namespace lkjai
