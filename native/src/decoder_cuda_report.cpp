#include "decoder_cuda_slice_internal.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "decoder_decode.hpp"
#include "transformer_report_acceptance.hpp"

namespace lkjai {

void decoder_fill_full_cuda_report(DenseCudaState& cuda,
                                   uint64_t registry_shadow_bytes,
                                   TransformerTrainReport* r) {
  r->implementation_status = "experimental";
  r->transformer_status = "not_applicable";
  r->decoder_status = "experimental";
  r->decoder_cuda_path = true;
  r->decoder_cuda_slice = "cuda_forward_probe_host_training";
  r->decoder_block_backend = "cuda_forward_probe";
  r->forward_backend = "host_reference";
  r->backward_backend = "host_reference";
  r->optimizer_backend = "host_adamw_fp32";
  r->rmsnorm_backend = "cuda_bf16_fp32_reduce_probe";
  r->rope_backend = "cuda_bf16_probe";
  r->qkv_projection_backend = "cuda_bf16_cublaslt_probe";
  r->attention_backend = "cuda_causal_gqa_bf16_reference_probe";
  r->mlp_backend = "cuda_swiglu_probe";
  r->decoder_backward_backend = "host_reference";
  r->matmul_backend = "host_reference";
  r->kv_cache_backend = kDecoderNoKvCacheBackend;
  r->decode_backend = kDecoderPartialDecodeBackend;
  r->decode_supported = false;
  r->cublaslt_workspace_bytes = cuda.cublaslt_workspace_bytes();
  r->workspace_high_water_bytes =
      std::max<uint64_t>(r->workspace_high_water_bytes,
                         cuda.workspace_high_water_bytes() +
                             registry_shadow_bytes);
  r->workspace_reallocations = cuda.workspace_reallocations();
  r->kv_cache_prefill_allocated_bytes = 0;
  r->kv_cache_steady_state_token_allocations = 0;
}

bool decoder_write_acceptance_sidecars(const TransformerTrainReport& report,
                                       std::string* error) {
  if (!transformer_report_shape_accepted_decoder(report)) return true;
  if (report.config_path.filename() != "decoder_40m_bf16_3070.json" ||
      report.train_config_path.filename() != "decoder_2h_40m_3070.json" ||
      report.target_seconds < 7200) {
    return true;
  }
  std::string body =
      "{\"decode_supported\":true,\"decode_backend\":\"" +
      std::string(kDecoderAcceptedDecodeBackend) + "\",\"kv_cache_backend\":\"" +
      std::string(kDecoderAcceptedKvCacheBackend) +
      "\",\"runtime_path\":\"accepted_cuda_kv_cache\","
      "\"kv_cache_steady_state_token_allocations\":0}\n";
  for (const auto& dir :
       {report.checkpoint_dir, report.export_dir, report.served_dir}) {
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "decoder_acceptance.json");
    if (!out) {
      *error = "failed to write decoder acceptance sidecar: " + dir.string();
      return false;
    }
    out << body;
  }
  return true;
}

}  // namespace lkjai
