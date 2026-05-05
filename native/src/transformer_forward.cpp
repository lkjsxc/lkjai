#include "transformer_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lkjai {
namespace {

using Vec = std::vector<float>;

void matvec(const Vec& x, const Parameter& w, Vec* y) {
  int in = w.shape[0];
  int out = w.shape[1];
  y->assign(static_cast<size_t>(out), 0.0f);
  for (int o = 0; o < out; ++o) {
    float sum = 0.0f;
    for (int i = 0; i < in; ++i) {
      sum += x[static_cast<size_t>(i)] * w.w[static_cast<size_t>(i * out + o)];
    }
    (*y)[static_cast<size_t>(o)] = sum;
  }
}

Vec rmsnorm(const Vec& x, const Parameter& weight, float eps) {
  float ss = 0.0f;
  for (float v : x) ss += v * v;
  float scale = 1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
  Vec y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * scale * weight.w[i];
  return y;
}

Vec attention(const std::vector<Vec>& q, const std::vector<Vec>& k,
              const std::vector<Vec>& v, const TransformerConfig& c, int pos) {
  Vec out(static_cast<size_t>(c.hidden_size), 0.0f);
  float inv_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
  for (int h = 0; h < c.heads; ++h) {
    int kvh = h % c.kv_heads;
    Vec scores(static_cast<size_t>(pos + 1));
    float mx = -INFINITY;
    for (int t = 0; t <= pos; ++t) {
      float s = 0.0f;
      for (int d = 0; d < c.head_dim; ++d) {
        s += q[pos][h * c.head_dim + d] * k[t][kvh * c.head_dim + d];
      }
      scores[static_cast<size_t>(t)] = s * inv_scale;
      mx = std::max(mx, scores[static_cast<size_t>(t)]);
    }
    float den = 0.0f;
    for (float& s : scores) {
      s = std::exp(s - mx);
      den += s;
    }
    for (int t = 0; t <= pos; ++t) {
      float p = scores[static_cast<size_t>(t)] / den;
      for (int d = 0; d < c.head_dim; ++d) {
        out[static_cast<size_t>(h * c.head_dim + d)] +=
            p * v[t][kvh * c.head_dim + d];
      }
    }
  }
  return out;
}

void apply_rope(Vec* x, int heads, int head_dim, int pos, float theta) {
  for (int h = 0; h < heads; ++h) {
    int base = h * head_dim;
    for (int d = 0; d + 1 < head_dim; d += 2) {
      double angle = static_cast<double>(pos) *
                     std::pow(static_cast<double>(theta),
                              -static_cast<double>(d) / head_dim);
      float cs = static_cast<float>(std::cos(angle));
      float sn = static_cast<float>(std::sin(angle));
      float a = (*x)[static_cast<size_t>(base + d)];
      float b = (*x)[static_cast<size_t>(base + d + 1)];
      (*x)[static_cast<size_t>(base + d)] = a * cs - b * sn;
      (*x)[static_cast<size_t>(base + d + 1)] = a * sn + b * cs;
    }
  }
}

void apply_layer(std::vector<Vec>* h, const TransformerLayer& l,
                 const TransformerConfig& c) {
  int seq = static_cast<int>(h->size());
  std::vector<Vec> q(static_cast<size_t>(seq)), k(static_cast<size_t>(seq));
  std::vector<Vec> v(static_cast<size_t>(seq)), norm(static_cast<size_t>(seq));
  for (int p = 0; p < seq; ++p) {
    norm[static_cast<size_t>(p)] = rmsnorm((*h)[p], l.attn_norm, c.rms_norm_eps);
    matvec(norm[static_cast<size_t>(p)], l.q_proj, &q[static_cast<size_t>(p)]);
    matvec(norm[static_cast<size_t>(p)], l.k_proj, &k[static_cast<size_t>(p)]);
    matvec(norm[static_cast<size_t>(p)], l.v_proj, &v[static_cast<size_t>(p)]);
    if (c.kind == "decoder") {
      apply_rope(&q[static_cast<size_t>(p)], c.heads, c.head_dim, p,
                 c.rope_theta);
      apply_rope(&k[static_cast<size_t>(p)], c.kv_heads, c.head_dim, p,
                 c.rope_theta);
    }
  }
  Vec tmp;
  for (int p = 0; p < seq; ++p) {
    auto ctx = attention(q, k, v, c, p);
    matvec(ctx, l.o_proj, &tmp);
    for (int i = 0; i < c.hidden_size; ++i) (*h)[p][i] += tmp[i];
    auto n = rmsnorm((*h)[p], l.mlp_norm, c.rms_norm_eps);
    Vec gate, up, down, act(static_cast<size_t>(c.ffn_size));
    matvec(n, l.gate_proj, &gate);
    matvec(n, l.up_proj, &up);
    for (int i = 0; i < c.ffn_size; ++i) {
      float s = 1.0f / (1.0f + std::exp(-gate[static_cast<size_t>(i)]));
      act[static_cast<size_t>(i)] = gate[static_cast<size_t>(i)] * s * up[i];
    }
    matvec(act, l.down_proj, &down);
    for (int i = 0; i < c.hidden_size; ++i) (*h)[p][i] += down[i];
  }
}

void logits_for(const Vec& h, const Parameter& head, Vec* logits) {
  int vocab = head.shape[0], hidden = head.shape[1];
  logits->assign(static_cast<size_t>(vocab), 0.0f);
  for (int v = 0; v < vocab; ++v) {
    float sum = 0.0f;
    for (int i = 0; i < hidden; ++i) sum += h[i] * head.w[v * hidden + i];
    (*logits)[static_cast<size_t>(v)] = sum;
  }
}

void ce_loss(const Vec& logits, int label, double* loss) {
  float mx = *std::max_element(logits.begin(), logits.end());
  float den = 0.0f;
  for (float v : logits) den += std::exp(v - mx);
  float p = std::exp(logits[static_cast<size_t>(label)] - mx) / den;
  *loss += -std::log(std::max(p, 1.0e-20f));
}

}  // namespace

ForwardResult transformer_forward(const PackedBatch& batch,
                                  const TransformerState& state) {
  const auto& c = state.cfg;
  ForwardResult out;
  for (int row = 0; row < batch.batch_size; ++row) {
    std::vector<Vec> h(static_cast<size_t>(batch.sequence_len));
    for (int p = 0; p < batch.sequence_len; ++p) {
      int tok = batch.tokens[row * batch.sequence_len + p] % c.vocab_size;
      auto base = state.tok_embeddings.w.begin() + tok * c.hidden_size;
      h[static_cast<size_t>(p)] = Vec(base, base + c.hidden_size);
      if (c.kind != "decoder") {
        auto pos = state.pos_embeddings.w.begin() + p * c.hidden_size;
        for (int i = 0; i < c.hidden_size; ++i) {
          h[static_cast<size_t>(p)][static_cast<size_t>(i)] += pos[i];
        }
      }
    }
    for (const auto& layer : state.layers) apply_layer(&h, layer, c);
    for (auto& x : h) x = rmsnorm(x, state.final_norm, c.rms_norm_eps);
    for (int p = 0; p + 1 < batch.sequence_len; ++p) {
      if (!batch.loss_mask[static_cast<size_t>(row * batch.sequence_len + p + 1)]) {
        continue;
      }
      int label = batch.tokens[row * batch.sequence_len + p + 1] % c.vocab_size;
      logits_for(h[static_cast<size_t>(p)], state.lm_head, &out.next_logits);
      ce_loss(out.next_logits, label, &out.loss);
      out.loss_logits = out.next_logits;
      out.loss_hidden = h[static_cast<size_t>(p)];
      out.loss_label = label;
      ++out.supervised;
    }
    out.last_hidden = h.back();
    logits_for(h.back(), state.lm_head, &out.next_logits);
  }
  if (out.supervised > 0) out.loss /= static_cast<double>(out.supervised);
  return out;
}

}  // namespace lkjai
