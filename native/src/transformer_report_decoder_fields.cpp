#include "transformer_report_decoder_fields.hpp"

#include <sstream>

#include "json_min.hpp"

namespace lkjai {

std::string transformer_decoder_runtime_report_json_fields(
    const TransformerTrainReport& report) {
  std::ostringstream out;
  out << ",\"optimizer_step_d2h_bytes\":"
      << static_cast<unsigned long long>(report.optimizer_step_d2h_bytes)
      << ",\"full_registry_d2h_bytes\":"
      << static_cast<unsigned long long>(report.full_registry_d2h_bytes)
      << ",\"decoder_parity_mode\":\""
      << json_escape(report.decoder_parity_mode) << "\""
      << ",\"decoder_parity_interval\":" << report.decoder_parity_interval
      << ",\"decoder_parity_sample_status\":\""
      << json_escape(report.decoder_parity_sample_status) << "\""
      << ",\"decoder_parity_sample_loss_diff\":"
      << report.decoder_parity_sample_loss_diff
      << ",\"decoder_parity_sample_logits_max_diff\":"
      << report.decoder_parity_sample_logits_max_diff
      << ",\"decoder_parity_sample_logits_mean_diff\":"
      << report.decoder_parity_sample_logits_mean_diff;
  return out.str();
}

}  // namespace lkjai
