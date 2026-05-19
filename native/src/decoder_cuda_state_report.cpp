#include "decoder_cuda_state.hpp"

#include <algorithm>

namespace lkjai {

void DecoderCudaState::fill_report(TransformerTrainReport* report) {
  decoder_fill_full_cuda_report(dense_cuda_, registry_shadow_bytes_, report);
  report->workspace_high_water_bytes =
      std::max<uint64_t>(report->workspace_high_water_bytes,
                         dense_cuda_.workspace_high_water_bytes() +
                             workspace_.high_water_bytes() +
                             registry_shadow_bytes_);
  report->optimizer_step_d2h_bytes = optimizer_step_d2h_bytes_;
  report->full_registry_d2h_bytes = full_registry_d2h_bytes_;
  report->decoder_parity_mode = parity_mode_;
  report->decoder_parity_interval = parity_interval_;
  report->decoder_parity_sample_status = parity_sample_status_;
  report->decoder_parity_sample_loss_diff = parity_sample_loss_diff_;
  report->decoder_parity_sample_logits_max_diff =
      parity_sample_logits_max_diff_;
  report->decoder_parity_sample_logits_mean_diff =
      parity_sample_logits_mean_diff_;
}

}  // namespace lkjai
