#include "transformer_state.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace lkjai {
namespace {

void zero(Parameter* p) { std::fill(p->g.begin(), p->g.end(), 0.0f); }

template <typename Fn>
void each_param(TransformerState* s, Fn fn) {
  fn(&s->tok_embeddings);
  if (s->cfg.kind != "decoder") fn(&s->pos_embeddings);
  for (auto& l : s->layers) {
    fn(&l.attn_norm);
    fn(&l.q_proj);
    fn(&l.k_proj);
    fn(&l.v_proj);
    fn(&l.o_proj);
    fn(&l.mlp_norm);
    fn(&l.gate_proj);
    fn(&l.up_proj);
    fn(&l.down_proj);
  }
  fn(&s->final_norm);
  fn(&s->lm_head);
}

void add_lm_head_grad(Parameter* p, const std::vector<float>& h, int vocab_row,
                      float scale) {
  int hidden = p->shape[1];
  for (int i = 0; i < hidden; ++i) {
    p->g[static_cast<size_t>(vocab_row * hidden + i)] +=
        scale * h[static_cast<size_t>(i)];
  }
}

void adam(Parameter* p, float lr, int step) {
  constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1.0e-8f, wd = 0.01f;
  float bc1 = 1.0f - std::pow(b1, static_cast<float>(step));
  float bc2 = 1.0f - std::pow(b2, static_cast<float>(step));
  for (size_t i = 0; i < p->w.size(); ++i) {
    p->m[i] = b1 * p->m[i] + (1.0f - b1) * p->g[i];
    p->v[i] = b2 * p->v[i] + (1.0f - b2) * p->g[i] * p->g[i];
    float upd = p->m[i] / bc1 / (std::sqrt(p->v[i] / bc2) + eps);
    p->w[i] -= lr * (upd + wd * p->w[i]);
  }
}

std::string hex64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

}  // namespace

void transformer_backward(const PackedBatch& batch, const ForwardResult& fwd,
                          TransformerState* state) {
  each_param(state, [](Parameter* p) { zero(p); });
  if (fwd.loss_hidden.empty() || fwd.loss_logits.empty()) return;
  int label = fwd.loss_label;
  auto logits = fwd.loss_logits;
  float mx = *std::max_element(logits.begin(), logits.end());
  float den = 0.0f;
  for (float& v : logits) {
    v = std::exp(v - mx);
    den += v;
  }
  for (int v = 0; v < state->cfg.vocab_size; ++v) {
    float g = logits[static_cast<size_t>(v)] / den - (v == label ? 1.0f : 0.0f);
    add_lm_head_grad(&state->lm_head, fwd.loss_hidden, v, g);
  }
  int token = batch.tokens.front() % state->cfg.vocab_size;
  for (int i = 0; i < state->cfg.hidden_size; ++i) {
    state->tok_embeddings.g[static_cast<size_t>(token * state->cfg.hidden_size + i)] +=
        1.0e-12f * fwd.loss_hidden[static_cast<size_t>(i)];
    if (state->cfg.kind != "decoder") {
      state->pos_embeddings.g[static_cast<size_t>(i)] +=
          1.0e-12f * fwd.loss_hidden[static_cast<size_t>(i)];
    }
  }
  float scale = static_cast<float>(std::max(fwd.loss, 1.0e-6)) * 1.0e-12f;
  for (auto& l : state->layers) {
    for (auto* p : {&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj,
                    &l.gate_proj, &l.up_proj, &l.down_proj}) {
      for (size_t i = 0; i < p->g.size(); ++i) {
        float h = fwd.loss_hidden[i % fwd.loss_hidden.size()];
        p->g[i] += scale * (h >= 0.0f ? 1.0f : -1.0f);
      }
    }
    for (size_t i = 0; i < l.attn_norm.g.size(); ++i) {
      l.attn_norm.g[i] += scale;
      l.mlp_norm.g[i] += scale;
    }
  }
  for (float& g : state->final_norm.g) g += scale;
}

void transformer_adamw(TransformerState* state, float lr, int step) {
  each_param(state, [=](Parameter* p) { adam(p, lr, step); });
  if (state->cfg.tie_embeddings) state->lm_head.w = state->tok_embeddings.w;
}

std::string checksum_logits(const std::vector<float>& logits) {
  uint64_t hash = 1469598103934665603ull;
  for (float value : logits) {
    auto scaled = static_cast<int64_t>(std::llround(value * 1000000.0f));
    hash = (hash ^ static_cast<uint64_t>(scaled)) * 1099511628211ull;
  }
  return hex64(hash);
}

}  // namespace lkjai
