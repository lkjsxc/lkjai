#include "train_report.hpp"
#include <cmath>
#include <sstream>
#include <vector>
#include "capability_json.hpp"
#include "json_min.hpp"
#include "train_report_digest.hpp"
#include "transformer_report_acceptance.hpp"
#include "transformer_report_decoder_fields.hpp"
#ifndef LKJAI_GIT_COMMIT
#define LKJAI_GIT_COMMIT "unknown"
#endif
#ifndef LKJAI_BUILD_TYPE
#define LKJAI_BUILD_TYPE "unknown"
#endif
#ifndef LKJAI_CUDA_ARCH_FLAGS
#define LKJAI_CUDA_ARCH_FLAGS "unknown"
#endif
namespace lkjai {
namespace {
void append_weight_part(std::ostringstream* out,
                        const DecoderWeightChangePart& part) {
  *out << "{\"max_abs_delta\":" << part.max_abs_delta
       << ",\"changed_elements\":"
       << static_cast<unsigned long long>(part.changed_elements)
       << ",\"changed_tensors\":" << part.changed_tensors << "}";
}

void append_transformer(std::ostringstream* out, const TransformerTrainReport& report,
                        const CudaStatus& cuda, const std::string& trainer_mode,
                        const std::string& status, const std::string& failure_reason) {
  auto kind = report.model_kind.empty() ? std::string("transformer") : report.model_kind;
  auto impl = report.implementation_status.empty()
                  ? std::string("experimental")
                  : report.implementation_status;
  auto decoder_status = report.decoder_status.empty()
      ? std::string(kind == "decoder" ? "experimental" : "not_applicable")
      : report.decoder_status;
  bool accepted_decoder = transformer_report_accepted_decoder(report);
  if (!accepted_decoder && impl == "accepted") impl = "experimental";
  double tokens_per_second = report.elapsed_seconds > 0.0
      ? static_cast<double>(report.input_tokens) / report.elapsed_seconds : 0.0;
  auto limitations = transformer_report_limitations(report, accepted_decoder);
  const auto& p = report.decoder_forward_probe;
  *out << "{\"schema\":\"lkjai-train-report\""
       << ",\"trainer_mode\":\"" << json_escape(trainer_mode) << "\""
       << ",\"mode\":\"" << json_escape(trainer_mode) << "\""
       << ",\"run_purpose\":\"" << json_escape(report.run_purpose) << "\""
       << ",\"model_kind\":\"" << json_escape(kind) << "\""
       << ",\"accepted_cuda_training\":"
       << (accepted_decoder ? "true" : "false")
       << ",\"implementation_status\":\"" << json_escape(impl) << "\""
       << ",\"transformer_status\":\""
       << json_escape(report.transformer_status) << "\""
       << ",\"decoder_status\":\"" << json_escape(decoder_status) << "\""
       << ",\"forward_backend\":\"" << json_escape(report.forward_backend)
       << "\",\"backward_backend\":\"" << json_escape(report.backward_backend)
       << "\",\"optimizer_backend\":\""
       << json_escape(report.optimizer_backend) << "\""
       << ",\"cuda_probe_passed\":"
       << (cuda_required_ok(cuda) ? "true" : "false")
       << ",\"status\":\"" << json_escape(status) << "\""
       << ",\"failure_reason\":\"" << json_escape(failure_reason) << "\""
       << ",\"limitations\":[";
  for (size_t i = 0; i < limitations.size(); ++i)
    *out << (i ? "," : "") << "\"" << json_escape(limitations[i]) << "\"";
  *out
       << "]"
       << ",\"precision_mode\":\"fp32-master-bf16-shadow-bf16-export\""
       << ",\"master_dtype\":\"f32\",\"shadow_dtype\":\"bf16\""
       << ",\"accumulation_dtype\":\"f32\",\"export_dtype\":\"bf16\""
       << ",\"dense_cuda_path\":false,\"transformer_cuda_path\":false"
       << ",\"decoder_cuda_path\":"
       << (report.decoder_cuda_path ? "true" : "false")
       << ",\"decode_supported\":"
       << (report.decode_supported ? "true" : "false") << ",\"embedding_tying\":\"" << json_escape(report.embedding_tying) << "\""
       << ",\"trainable_tensor_count\":" << report.trainable_tensor_count
       << ",\"decoder_cuda_slice\":\""
       << json_escape(report.decoder_cuda_slice) << "\""
       << ",\"decoder_block_backend\":\""
       << json_escape(report.decoder_block_backend) << "\""
       << ",\"decoder_block_forward_in_training\":" << (report.decoder_block_forward_in_training ? "true" : "false") << ",\"decoder_block_forward_steps\":" << report.decoder_block_forward_steps
       << ",\"decoder_forward_probe\":{\"status\":\"" << json_escape(p.status)
       << "\",\"recorded\":" << (p.recorded ? "true" : "false")
       << ",\"rmsnorm\":" << (p.rmsnorm ? "true" : "false")
       << ",\"rope\":" << (p.rope ? "true" : "false")
       << ",\"qkv_projection\":" << (p.qkv_projection ? "true" : "false")
       << ",\"attention\":" << (p.attention ? "true" : "false")
       << ",\"output_projection\":" << (p.output_projection ? "true" : "false")
       << ",\"attention_residual\":" << (p.attention_residual ? "true" : "false")
       << ",\"mlp_norm\":" << (p.mlp_norm ? "true" : "false")
       << ",\"swiglu\":" << (p.swiglu ? "true" : "false")
       << ",\"down_projection\":" << (p.down_projection ? "true" : "false")
       << ",\"block_residual\":" << (p.block_residual ? "true" : "false")
       << ",\"output_finite\":" << (p.output_finite ? "true" : "false")
       << ",\"batch\":" << p.batch << ",\"sequence\":" << p.sequence
       << ",\"output_rows\":" << p.output_rows
       << ",\"output_hidden_size\":" << p.output_hidden_size
       << ",\"workspace_bytes\":" << static_cast<unsigned long long>(p.workspace_bytes)
       << "}"
       << ",\"rmsnorm_backend\":\"" << json_escape(report.rmsnorm_backend)
       << "\",\"rope_backend\":\"" << json_escape(report.rope_backend)
       << "\",\"qkv_projection_backend\":\"" << json_escape(report.qkv_projection_backend) << "\""
       << ",\"attention_backend\":\"" << json_escape(report.attention_backend) << "\""
       << ",\"mlp_backend\":\"" << json_escape(report.mlp_backend)
       << "\",\"decoder_backward_backend\":\"" << json_escape(!accepted_decoder && report.decoder_backward_backend == "cuda_full_decoder" ? "not_accepted_cuda_full_decoder" : report.decoder_backward_backend) << "\""
       << ",\"decoder_gradient_source\":\"" << json_escape(report.decoder_gradient_source) << "\""
       << ",\"matmul_backend\":\"" << json_escape(report.matmul_backend)
       << "\",\"kv_cache_backend\":\"" << json_escape(report.kv_cache_backend)
       << "\",\"decode_backend\":\"" << json_escape(!accepted_decoder && report.decode_backend == "cuda_kv_cache" ? "cuda_reference_kv_cache" : report.decode_backend)
       << "\",\"cublaslt_workspace_bytes\":" << static_cast<unsigned long long>(report.cublaslt_workspace_bytes)
       << ",\"workspace_high_water_bytes\":" << static_cast<unsigned long long>(report.workspace_high_water_bytes)
       << ",\"workspace_reallocations\":" << report.workspace_reallocations
       << ",\"kv_cache_prefill_allocated_bytes\":"
       << static_cast<unsigned long long>(report.kv_cache_prefill_allocated_bytes)
       << ",\"kv_cache_steady_state_token_allocations\":"
       << report.kv_cache_steady_state_token_allocations
       << ",\"transformer_cuda_probe\":true"
       << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
       << ",\"cuda_device_name\":\"" << json_escape(cuda.device) << "\""
       << ",\"cuda_driver_version\":" << cuda.cuda_driver_version << ",\"cuda_runtime_version\":" << cuda.cuda_runtime_version
       << ",\"cudnn_version\":" << cuda.cudnn_version << ",\"cuda_device_count\":" << cuda.device_count
       << ",\"cuda_device_index\":" << cuda.device_index
       << ",\"cuda_total_global_memory\":" << static_cast<unsigned long long>(cuda.total_global_memory)
       << ",\"cuda_sm_count\":" << cuda.sm_count << ",\"cuda_arch_flags\":\"" << json_escape(LKJAI_CUDA_ARCH_FLAGS)
       << "\",\"git_commit\":\"" << json_escape(LKJAI_GIT_COMMIT) << "\""
       << ",\"build_type\":\"" << json_escape(LKJAI_BUILD_TYPE) << "\""
       << ",\"config_path\":\"" << json_escape(report.config_path.string()) << "\",\"config_digest\":\"" << train_report_file_digest(report.config_path) << "\""
       << ",\"dataset_path\":\"" << json_escape(report.packed_cache.string())
       << "\",\"packed_cache_path\":\"" << json_escape(report.packed_cache.string()) << "\""
       << ",\"dataset_digest\":\"" << train_report_packed_cache_digest(report.packed_cache)
       << "\",\"train_config_path\":\"" << json_escape(report.train_config_path.string()) << "\""
       << ",\"seed\":" << json_int_value(read_text(report.config_path), "seed", 0)
       << ",\"batch_size\":" << report.batch_size
       << ",\"seq_len\":" << report.seq_len
       << ",\"grad_accum\":" << report.grad_accum
       << ",\"layers\":" << report.layers << ",\"heads\":" << report.heads
       << ",\"kv_heads\":" << report.kv_heads
       << ",\"hidden_size\":" << report.hidden_size
       << ",\"head_dim\":" << report.head_dim
       << ",\"ffn_size\":" << report.ffn_size
       << ",\"context\":" << report.context
       << ",\"parameter_count\":" << report.parameter_count
       << ",\"target_seconds\":" << report.target_seconds
       << ",\"deadline_hit\":" << (report.deadline_hit ? "true" : "false")
       << ",\"stop_reason\":\"" << json_escape(report.stop_reason) << "\""
       << ",\"tokens_seen\":" << report.input_tokens
       << ",\"input_tokens\":" << report.input_tokens
       << ",\"loss_tokens\":" << report.loss_tokens
       << ",\"optimizer_steps\":" << report.steps
       << ",\"steps\":" << report.steps
       << ",\"start_step\":" << report.start_step
       << ",\"microsteps\":" << report.microsteps
       << ",\"initial_loss\":" << report.initial_loss
       << ",\"loss\":" << report.loss << ",\"loss_finite\":" << (std::isfinite(report.loss) ? "true" : "false")
       << ",\"weight_changed\":" << (report.trainable_weight_changed ? "true" : "false")
       << ",\"trainable_weight_changed\":" << (report.trainable_weight_changed ? "true" : "false")
       << ",\"embedding_weight_changed\":" << (report.embedding_weight_changed ? "true" : "false")
       << ",\"lm_head_weight_changed\":" << (report.lm_head_weight_changed ? "true" : "false")
       << ",\"non_embedding_weight_changed\":" << (report.non_embedding_weight_changed ? "true" : "false")
       << ",\"decoder_block_weight_changed\":" << (report.decoder_block_weight_changed ? "true" : "false")
       << ",\"decoder_weight_change\":{\"embedding\":";
  append_weight_part(out, report.decoder_weight_change.embedding);
  *out << ",\"lm_head\":";
  append_weight_part(out, report.decoder_weight_change.lm_head);
  *out << ",\"non_embedding\":";
  append_weight_part(out, report.decoder_weight_change.non_embedding);
  *out << ",\"decoder_block\":";
  append_weight_part(out, report.decoder_weight_change.decoder_block);
  *out << ",\"changed_tensors\":"
       << report.decoder_weight_change.changed_tensors << "}"
       << ",\"elapsed_ms\":" << report.elapsed_seconds * 1000.0
       << ",\"elapsed_seconds\":" << report.elapsed_seconds
       << ",\"tokens_per_second\":" << tokens_per_second
       << ",\"checkpoint_path\":\""
       << json_escape(report.checkpoint_dir.string()) << "\""
       << ",\"checkpoint_checksum\":\""
       << train_report_manifest_checksum(report.checkpoint_dir) << "\""
       << ",\"export_path\":\"" << json_escape(report.export_dir.string())
       << "\",\"export_checksum\":\"" << train_report_manifest_checksum(report.export_dir)
       << "\",\"served_path\":\"" << json_escape(report.served_dir.string())
       << "\",\"logits_checksum\":\""
       << json_escape(report.logits_check_checksum) << "\""
       << ",\"logits_check_passed\":" << (report.logits_check_passed ? "true" : "false")
       << ",\"logits_check\":"
       << (report.logits_check_json.empty()
               ? "{\"status\":\"fail\",\"validation_target\":"
                 "\"exported_bf16_weights\",\"checksum\":\"\"}"
               : report.logits_check_json)
       << ",\"timings\":{\"batch_load\":" << report.batch_load_seconds
       << ",\"h2d\":" << report.h2d_seconds
       << ",\"forward\":" << report.forward_seconds
       << ",\"backward\":" << report.backward_seconds
       << ",\"optimizer\":" << report.optimizer_seconds
       << ",\"checkpoint\":" << report.checkpoint_export_seconds
       << ",\"export\":" << report.export_seconds << "}"
       << ",\"capability\":{" << capability_json_fields(cuda) << "}";
} }  // namespace
std::string transformer_train_report_json(const TransformerTrainReport& report, const CudaStatus& cuda, const std::string& trainer_mode, const std::string& status, const std::string& failure_reason) { std::ostringstream out; append_transformer(&out, report, cuda, trainer_mode, status, failure_reason); out << transformer_decoder_runtime_report_json_fields(report) << "}"; return out.str(); }
}  // namespace lkjai
