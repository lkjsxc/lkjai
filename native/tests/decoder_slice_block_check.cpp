#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "decoder_cuda_slice_internal.hpp"
#include "transformer_state.hpp"

namespace {

using Vec = std::vector<float>;

void matvec(const Vec& x, const lkjai::Parameter& w, Vec* y) {
  int in = w.shape[0], out = w.shape[1];
  y->assign(static_cast<size_t>(out), 0.0f);
  for (int o = 0; o < out; ++o) {
    for (int i = 0; i < in; ++i) {
      (*y)[static_cast<size_t>(o)] +=
          x[static_cast<size_t>(i)] * w.w[static_cast<size_t>(i * out + o)];
    }
  }
}

Vec rmsnorm(const Vec& x, const lkjai::Parameter& w, float eps) {
  float ss = 0.0f;
  for (float v : x) ss += v * v;
  float scale = 1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
  Vec y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * scale * w.w[i];
  return y;
}

void rope(Vec* x, int heads, int head_dim, int pos, float theta) {
  for (int h = 0; h < heads; ++h) {
    int base = h * head_dim;
    for (int d = 0; d + 1 < head_dim; d += 2) {
      double angle = double(pos) * std::pow(double(theta), -double(d) / head_dim);
      float c = static_cast<float>(std::cos(angle));
      float s = static_cast<float>(std::sin(angle));
      float a = (*x)[static_cast<size_t>(base + d)];
      float b = (*x)[static_cast<size_t>(base + d + 1)];
      (*x)[static_cast<size_t>(base + d)] = a * c - b * s;
      (*x)[static_cast<size_t>(base + d + 1)] = a * s + b * c;
    }
  }
}

Vec attention(const std::vector<Vec>& q, const std::vector<Vec>& k,
              const std::vector<Vec>& v, const lkjai::TransformerConfig& c,
              int pos) {
  Vec out(static_cast<size_t>(c.hidden_size), 0.0f);
  float inv = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
  for (int h = 0; h < c.heads; ++h) {
    int kvh = h % c.kv_heads;
    Vec score(static_cast<size_t>(pos + 1));
    float mx = -INFINITY;
    for (int t = 0; t <= pos; ++t) {
      for (int d = 0; d < c.head_dim; ++d)
        score[static_cast<size_t>(t)] +=
            q[static_cast<size_t>(pos)][h * c.head_dim + d] *
            k[static_cast<size_t>(t)][kvh * c.head_dim + d];
      score[static_cast<size_t>(t)] *= inv;
      mx = std::max(mx, score[static_cast<size_t>(t)]);
    }
    float den = 0.0f;
    for (float& s : score) {
      s = std::exp(s - mx);
      den += s;
    }
    for (int t = 0; t <= pos; ++t) {
      float p = score[static_cast<size_t>(t)] / den;
      for (int d = 0; d < c.head_dim; ++d)
        out[static_cast<size_t>(h * c.head_dim + d)] +=
            p * v[static_cast<size_t>(t)][kvh * c.head_dim + d];
    }
  }
  return out;
}

Vec host_block_output(const lkjai::TransformerState& state,
                      const lkjai::PackedBatch& batch) {
  const auto& c = state.cfg;
  const auto& l = state.layers.front();
  std::vector<Vec> h(static_cast<size_t>(batch.sequence_len));
  for (int p = 0; p < batch.sequence_len; ++p) {
    int tok = batch.tokens[static_cast<size_t>(p)] % c.vocab_size;
    auto base = state.tok_embeddings.w.begin() + tok * c.hidden_size;
    h[static_cast<size_t>(p)] = Vec(base, base + c.hidden_size);
  }
  std::vector<Vec> q(h.size()), k(h.size()), v(h.size());
  for (int p = 0; p < batch.sequence_len; ++p) {
    auto n = rmsnorm(h[static_cast<size_t>(p)], l.attn_norm, c.rms_norm_eps);
    matvec(n, l.q_proj, &q[static_cast<size_t>(p)]);
    matvec(n, l.k_proj, &k[static_cast<size_t>(p)]);
    matvec(n, l.v_proj, &v[static_cast<size_t>(p)]);
    rope(&q[static_cast<size_t>(p)], c.heads, c.head_dim, p, c.rope_theta);
    rope(&k[static_cast<size_t>(p)], c.kv_heads, c.head_dim, p, c.rope_theta);
  }
  Vec tmp;
  for (int p = 0; p < batch.sequence_len; ++p) {
    auto ctx = attention(q, k, v, c, p);
    matvec(ctx, l.o_proj, &tmp);
    for (int i = 0; i < c.hidden_size; ++i) h[static_cast<size_t>(p)][i] += tmp[i];
    auto n = rmsnorm(h[static_cast<size_t>(p)], l.mlp_norm, c.rms_norm_eps);
    Vec gate, up, down, act(static_cast<size_t>(c.ffn_size));
    matvec(n, l.gate_proj, &gate);
    matvec(n, l.up_proj, &up);
    for (int i = 0; i < c.ffn_size; ++i) {
      float g = gate[static_cast<size_t>(i)];
      act[static_cast<size_t>(i)] = (g / (1.0f + std::exp(-g))) * up[i];
    }
    matvec(act, l.down_proj, &down);
    for (int i = 0; i < c.hidden_size; ++i) h[static_cast<size_t>(p)][i] += down[i];
  }
  Vec flat;
  for (const auto& row : h) flat.insert(flat.end(), row.begin(), row.end());
  return flat;
}

bool close(const Vec& got, const Vec& want) {
  if (got.size() != want.size()) return false;
  double sum = 0.0;
  float max_diff = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) return false;
    float diff = std::fabs(got[i] - want[i]);
    max_diff = std::max(max_diff, diff);
    sum += diff;
  }
  double mean = sum / static_cast<double>(got.size());
  if (max_diff <= 0.05f && mean <= 0.01) return true;
  std::cerr << "block parity max_abs=" << max_diff << " mean_abs=" << mean << "\n";
  return false;
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable: "
              << (cuda.error.empty() ? cuda.warning : cuda.error) << "\n";
    return 1;
  }
  lkjai::TransformerConfig cfg;
  cfg.kind = "decoder";
  cfg.vocab_size = 64;
  cfg.context = 8;
  cfg.layers = 1;
  cfg.hidden_size = 32;
  cfg.heads = 4;
  cfg.kv_heads = 4;
  cfg.head_dim = 8;
  cfg.ffn_size = 64;
  cfg.tie_embeddings = true;
  lkjai::TransformerState state;
  lkjai::init_transformer_state(cfg, &state);
  lkjai::PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 4;
  batch.tokens = {1, 2, 3, 4};
  batch.loss_mask = {1, 1, 1, 1};
  lkjai::DecoderCudaForwardSubstrateReport report;
  std::string error;
  if (!lkjai::decoder_cuda_slice_run_block_forward(state, batch, &report,
                                                   &error)) {
    std::cerr << error << "\n";
    return 2;
  }
  bool ok = report.outputs_finite && report.rmsnorm_checked &&
            report.qkv_projection_checked && report.rope_checked &&
            report.attention_checked && report.o_projection_checked &&
            report.attention_residual_checked && report.mlp_norm_checked &&
            report.swiglu_checked && report.down_projection_checked &&
            report.block_residual_checked && report.probe_batch == 1 &&
            report.probe_seq == 4 && report.output_rows == 4 &&
            report.output_hidden_size == cfg.hidden_size &&
            report.output_hidden.size() == 4ull * cfg.hidden_size;
  if (!ok) {
    std::cerr << "decoder slice block report missing expected checks\n";
    return 3;
  }
  if (!close(report.output_hidden, host_block_output(state, batch))) {
    std::cerr << "decoder slice block output did not match host reference\n";
    return 4;
  }
  return 0;
}
