#include <iostream>
#include <string>

#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "decoder_cuda_slice_internal.hpp"
#include "transformer_state.hpp"

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
            report.probe_seq == 4;
  if (!ok) {
    std::cerr << "decoder slice block report missing expected checks\n";
    return 3;
  }
  return 0;
}
