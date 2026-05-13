#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_state.hpp"
#include "transformer_state.hpp"

namespace {

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double out = 0.0;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    out = std::max(out, std::abs(static_cast<double>(a[i]) - b[i]));
  }
  return out;
}

double mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  size_t n = std::min(a.size(), b.size());
  if (n == 0) return 0.0;
  double out = 0.0;
  for (size_t i = 0; i < n; ++i) {
    out += std::abs(static_cast<double>(a[i]) - b[i]);
  }
  return out / static_cast<double>(n);
}

bool finite_all(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable: "
              << (cuda.error.empty() ? cuda.warning : cuda.error) << "\n";
    return 1;
  }
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  lkjai::TransformerConfig cfg;
  std::string error;
  if (!lkjai::load_transformer_config(
          repo / "configs" / "native" / "decoder_debug_bf16.json", &cfg,
          &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  lkjai::TransformerState state;
  lkjai::init_transformer_state(cfg, &state);
  lkjai::PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 4;
  batch.tokens = {1, 7, 11, 19};
  batch.loss_mask = {1, 1, 1, 1};
  auto host = lkjai::transformer_forward(batch, state);
  lkjai::DecoderCudaState cuda_state(state.cfg, state);
  std::vector<float> logits;
  double loss = cuda_state.forward_backward(batch, &logits, nullptr, nullptr,
                                            nullptr, 1.0f, true);
  double loss_diff = std::abs(loss - host.loss);
  double logits_max = max_abs_diff(logits, host.loss_logits);
  double logits_mean = mean_abs_diff(logits, host.loss_logits);
  if (!std::isfinite(loss) || loss_diff > 0.15 || logits_max > 0.08 ||
      logits_mean > 0.025 || !finite_all(logits)) {
    std::cerr << "decoder CUDA train-forward parity failed loss_diff="
              << loss_diff << " logits_max=" << logits_max
              << " logits_mean=" << logits_mean << "\n";
    return 1;
  }

  auto zero = batch;
  zero.loss_mask = {0, 0, 0, 0};
  logits.clear();
  double zero_loss = cuda_state.forward_backward(zero, &logits, nullptr,
                                                 nullptr, nullptr, 1.0f, true);
  auto grad_logits = cuda_state.debug_last_grad_logits();
  bool grad_zero = std::all_of(grad_logits.begin(), grad_logits.end(),
                               [](float value) { return value == 0.0f; });
  bool logits_zero = std::all_of(logits.begin(), logits.end(),
                                 [](float value) { return value == 0.0f; });
  if (!std::isfinite(zero_loss) || zero_loss != 0.0 || !grad_zero ||
      !logits_zero) {
    std::cerr << "decoder CUDA zero-supervision path failed\n";
    return 1;
  }

  std::cout << "{\"status\":\"pass\",\"decoder_cuda_train_forward\":true"
            << ",\"loss_diff\":" << loss_diff
            << ",\"logits_max_abs\":" << logits_max
            << ",\"logits_mean_abs\":" << logits_mean << "}\n";
  return 0;
}
