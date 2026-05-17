#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "cuda_probe.hpp"
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
  float before = state.layers[0].q_proj.w[0];
  lkjai::DecoderCudaState cuda_state(state.cfg, state);
  lkjai::PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 4;
  batch.tokens = {1, 7, 11, 19};
  batch.loss_mask = {1, 1, 1, 1};
  cuda_state.forward_backward(batch, nullptr, nullptr, nullptr, nullptr, 1.0f,
                              true);
  cuda_state.optimizer_step(1.0e-3f, 1);
  auto updated = cuda_state.copy_to_host();
  float after = updated.layers[0].q_proj.w[0];
  if (!std::isfinite(after) || after == before) {
    std::cerr << "decoder block weight did not change\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"decoder_block_weight_changed\":true}\n";
  return 0;
}
