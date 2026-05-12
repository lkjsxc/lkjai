#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "transformer_state.hpp"

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
  lkjai::DecoderCudaFullForwardReport report;
  if (!lkjai::decoder_cuda_full_forward_probe(state, batch, &report, &error)) {
    std::cerr << error << " hidden_max_abs=" << report.hidden_max_abs
              << " logits_max_abs=" << report.logits_max_abs << "\n";
    return 1;
  }
  if (!report.layers_checked || !report.final_norm_checked ||
      !report.logits_checked || !report.hidden_close ||
      !report.logits_close || report.workspace_bytes == 0 ||
      report.layers != cfg.layers || report.sequence != batch.sequence_len) {
    std::cerr << "decoder full forward report missing expected checks\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"decoder_full_forward\":true"
            << ",\"hidden_max_abs\":" << report.hidden_max_abs
            << ",\"logits_max_abs\":" << report.logits_max_abs << "}\n";
  return 0;
}
