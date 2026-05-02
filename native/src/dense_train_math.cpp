#include "dense_train_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>

namespace lkjai {

void init_dense_state(const DenseConfig& cfg, DenseTrainState* state) {
  state->cfg = cfg;
  auto count = static_cast<size_t>(cfg.vocab_size * cfg.hidden_size);
  state->emb.resize(count);
  state->head.resize(count);
  std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
  std::normal_distribution<float> dist(0.0f, 0.02f);
  for (float& value : state->emb) value = dist(rng);
  for (float& value : state->head) value = dist(rng);
  state->grad_emb.resize(count);
  state->grad_head.resize(count);
  state->m_emb.resize(count);
  state->v_emb.resize(count);
  state->m_head.resize(count);
  state->v_head.resize(count);
}

double dense_forward_backward(const PackedBatch& batch,
                              DenseTrainState* state) {
  const auto& cfg = state->cfg;
  std::fill(state->grad_emb.begin(), state->grad_emb.end(), 0.0f);
  std::fill(state->grad_head.begin(), state->grad_head.end(), 0.0f);
  std::vector<float> logits(static_cast<size_t>(cfg.vocab_size));
  std::vector<float> grad_h(static_cast<size_t>(cfg.hidden_size));
  double loss = 0.0;
  int seen = 0;
  for (int row = 0; row < batch.batch_size; ++row) {
    for (int pos = 0; pos + 1 < batch.sequence_len; ++pos) {
      auto base = static_cast<size_t>(row * batch.sequence_len + pos);
      if (batch.loss_mask[base + 1] == 0) continue;
      int token = batch.tokens[base] % cfg.vocab_size;
      int label = batch.tokens[base + 1] % cfg.vocab_size;
      auto h = state->emb.data() + static_cast<size_t>(token * cfg.hidden_size);
      float max_logit = -INFINITY;
      for (int v = 0; v < cfg.vocab_size; ++v) {
        float sum = 0.0f;
        auto w = state->head.data() + static_cast<size_t>(v * cfg.hidden_size);
        for (int i = 0; i < cfg.hidden_size; ++i) sum += h[i] * w[i];
        logits[v] = sum;
        max_logit = std::max(max_logit, sum);
      }
      float denom = 0.0f;
      for (float& logit : logits) {
        logit = std::exp(logit - max_logit);
        denom += logit;
      }
      loss += -std::log(std::max(logits[label] / denom, 1.0e-20f));
      std::fill(grad_h.begin(), grad_h.end(), 0.0f);
      for (int v = 0; v < cfg.vocab_size; ++v) {
        float grad = logits[v] / denom - (v == label ? 1.0f : 0.0f);
        auto gh = state->grad_head.data() + static_cast<size_t>(v * cfg.hidden_size);
        auto w = state->head.data() + static_cast<size_t>(v * cfg.hidden_size);
        for (int i = 0; i < cfg.hidden_size; ++i) {
          gh[i] += grad * h[i];
          grad_h[i] += grad * w[i];
        }
      }
      auto ge = state->grad_emb.data() + static_cast<size_t>(token * cfg.hidden_size);
      for (int i = 0; i < cfg.hidden_size; ++i) ge[i] += grad_h[i];
      ++seen;
    }
  }
  float scale = seen > 0 ? 1.0f / static_cast<float>(seen) : 1.0f;
  for (float& value : state->grad_emb) value *= scale;
  for (float& value : state->grad_head) value *= scale;
  return seen > 0 ? loss / static_cast<double>(seen) : 0.0;
}

void dense_adamw(std::vector<float>* weight, std::vector<float>* m,
                 std::vector<float>* v, const std::vector<float>& grad,
                 float lr, int step) {
  constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1.0e-8f, wd = 0.01f;
  float bc1 = 1.0f - std::pow(b1, static_cast<float>(step));
  float bc2 = 1.0f - std::pow(b2, static_cast<float>(step));
  for (size_t i = 0; i < weight->size(); ++i) {
    (*m)[i] = b1 * (*m)[i] + (1.0f - b1) * grad[i];
    (*v)[i] = b2 * (*v)[i] + (1.0f - b2) * grad[i] * grad[i];
    float update = (*m)[i] / bc1 / (std::sqrt((*v)[i] / bc2) + eps);
    (*weight)[i] -= lr * (update + wd * (*weight)[i]);
  }
}

}  // namespace lkjai
