#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_block_check_ref.hpp"
#include "decoder_cuda_state.hpp"
#include "transformer_state.hpp"

namespace {

bool load_debug_config(lkjai::TransformerConfig* cfg, std::string* error) {
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  return lkjai::load_transformer_config(
      repo / "configs" / "native" / "decoder_debug_bf16.json", cfg, error);
}

std::vector<float> bf16_round(const std::vector<float>& values) {
  std::vector<float> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = f32(bf16(values[i]));
  return out;
}

std::vector<float> cpu_lm_head_grad(const std::vector<float>& grad_logits,
                                    const std::vector<float>& final_norm,
                                    int rows, int vocab, int hidden) {
  std::vector<float> out(static_cast<size_t>(vocab) * hidden);
  for (int v = 0; v < vocab; ++v) {
    for (int h = 0; h < hidden; ++h) {
      float sum = 0.0f;
      for (int r = 0; r < rows; ++r) {
        sum += grad_logits[static_cast<size_t>(r) * vocab + v] *
               final_norm[static_cast<size_t>(r) * hidden + h];
      }
      out[static_cast<size_t>(v) * hidden + h] = sum;
    }
  }
  return out;
}

std::vector<float> cpu_lm_head_dhidden(const std::vector<float>& grad_logits,
                                       const std::vector<float>& lm_head,
                                       int rows, int vocab, int hidden) {
  std::vector<float> out(static_cast<size_t>(rows) * hidden);
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < hidden; ++h) {
      float sum = 0.0f;
      for (int v = 0; v < vocab; ++v) {
        sum += grad_logits[static_cast<size_t>(r) * vocab + v] *
               lm_head[static_cast<size_t>(v) * hidden + h];
      }
      out[static_cast<size_t>(r) * hidden + h] = sum;
    }
  }
  return out;
}

void cpu_rmsnorm_backward(const std::vector<float>& input,
                          const std::vector<float>& weight,
                          const std::vector<float>& d_output, int rows,
                          int hidden, float eps, std::vector<float>* d_input,
                          std::vector<float>* d_weight) {
  d_input->assign(static_cast<size_t>(rows) * hidden, 0.0f);
  d_weight->assign(static_cast<size_t>(hidden), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    double dot = 0.0;
    for (int h = 0; h < hidden; ++h) {
      size_t i = static_cast<size_t>(r) * hidden + h;
      float xv = input[i];
      float dy = d_output[i];
      ss += static_cast<double>(xv) * xv;
      dot += static_cast<double>(dy) * weight[static_cast<size_t>(h)] * xv;
    }
    float inv =
        1.0f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    float coeff = inv * inv * inv * static_cast<float>(dot) /
                  static_cast<float>(hidden);
    for (int h = 0; h < hidden; ++h) {
      size_t i = static_cast<size_t>(r) * hidden + h;
      float xv = input[i];
      float dy = d_output[i];
      (*d_input)[i] = dy * weight[static_cast<size_t>(h)] * inv - xv * coeff;
      (*d_weight)[static_cast<size_t>(h)] += dy * xv * inv;
    }
  }
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable\n";
    return 1;
  }
  lkjai::TransformerConfig cfg;
  std::string error;
  if (!load_debug_config(&cfg, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (cfg.tie_embeddings) {
    std::cerr << "decoder final norm backward check requires untied head\n";
    return 1;
  }
  lkjai::TransformerState state;
  lkjai::init_transformer_state(cfg, &state);
  lkjai::PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 4;
  batch.tokens = {1, 7, 11, 19};
  batch.loss_mask = {0, 0, 1, 0};

  lkjai::DecoderCudaState cuda_state(state.cfg, state);
  double loss = cuda_state.forward_backward(batch, nullptr, nullptr, nullptr,
                                            nullptr, 1.0f, true);
  if (!std::isfinite(loss)) {
    std::cerr << "loss is non-finite\n";
    return 1;
  }

  auto grad_logits = cuda_state.debug_last_grad_logits();
  auto final_norm_input = cuda_state.debug_last_final_norm_input();
  auto final_norm = cuda_state.debug_last_final_norm();
  auto grad_final_norm = cuda_state.debug_last_grad_final_norm();
  auto grad_final_norm_input = cuda_state.debug_last_grad_final_norm_input();
  auto with_grad = cuda_state.copy_to_host();

  int rows = batch.batch_size * batch.sequence_len;
  int vocab = cfg.vocab_size;
  int hidden = cfg.hidden_size;
  auto want_lm_head =
      cpu_lm_head_grad(grad_logits, final_norm, rows, vocab, hidden);
  auto want_grad_final_norm = cpu_lm_head_dhidden(
      grad_logits, bf16_round(state.lm_head.w), rows, vocab, hidden);
  std::vector<float> want_norm_input;
  std::vector<float> want_norm_weight;
  cpu_rmsnorm_backward(final_norm_input, state.final_norm.w, grad_final_norm,
                       rows, hidden, cfg.rms_norm_eps, &want_norm_input,
                       &want_norm_weight);

  bool ok = close_enough(with_grad.lm_head.g, want_lm_head, 0.006, 0.002,
                         "LM-head dW") &&
            close_enough(grad_final_norm, want_grad_final_norm, 0.006, 0.002,
                         "LM-head dHidden") &&
            close_enough(with_grad.final_norm.g, want_norm_weight, 0.02,
                         0.006, "final RMSNorm dWeight") &&
            close_enough(grad_final_norm_input, want_norm_input, 0.006,
                         0.002, "final RMSNorm dInput");
  if (!ok) return 1;
  std::cout
      << "{\"status\":\"pass\",\"decoder_cuda_lm_head_final_norm_backward\":true}\n";
  return 0;
}
