#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "cuda_probe.hpp"
#include "decoder_cuda_state.hpp"
#include "transformer_state.hpp"

namespace {

bool any_nonzero(const std::vector<float>& values) {
  return std::any_of(values.begin(), values.end(),
                     [](float value) { return value != 0.0f; });
}

bool load_debug_config(lkjai::TransformerConfig* cfg, std::string* error) {
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  return lkjai::load_transformer_config(
      repo / "configs" / "native" / "decoder_debug_bf16.json", cfg, error);
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
  lkjai::TransformerState state;
  lkjai::init_transformer_state(cfg, &state);
  lkjai::PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 4;
  batch.tokens = {1, 7, 11, 19};
  batch.loss_mask = {1, 1, 1, 1};

  lkjai::DecoderCudaState cuda_state(state.cfg, state);
  std::vector<float> logits;
  double loss = cuda_state.forward_backward(batch, &logits, nullptr, nullptr,
                                            nullptr, 1.0f, true);
  auto with_grad = cuda_state.copy_to_host();
  bool ok = std::isfinite(loss) && any_nonzero(with_grad.lm_head.g) &&
            any_nonzero(with_grad.tok_embeddings.g) &&
            any_nonzero(with_grad.layers[0].q_proj.g) &&
            any_nonzero(with_grad.final_norm.g);
  if (!ok) {
    std::cerr << "device-origin decoder gradients were not populated\n";
    return 1;
  }

  auto zero = batch;
  zero.loss_mask = {0, 0, 0, 0};
  cuda_state.forward_backward(zero, nullptr, nullptr, nullptr, nullptr, 1.0f,
                              true);
  auto cleared = cuda_state.copy_to_host();
  if (any_nonzero(cleared.lm_head.g) ||
      any_nonzero(cleared.final_norm.g) ||
      any_nonzero(cleared.layers[0].q_proj.g)) {
    std::cerr << "reset_grads did not clear decoder CUDA gradients\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"decoder_cuda_full_backward\":true}\n";
  return 0;
}
