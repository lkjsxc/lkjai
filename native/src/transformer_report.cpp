#include "train_report.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include "artifact.hpp"
#include "capability_json.hpp"
#include "decoder_decode.hpp"
#include "json_min.hpp"
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
std::string file_digest(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  uint64_t hash = 1469598103934665603ull;
  char ch = 0;
  while (in.get(ch)) {
    hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}
std::string packed_cache_digest(const std::filesystem::path& dir) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& name :
       {"metadata.json", "tokens.bin", "loss_mask.bin", "starts.bin"}) {
    auto path = dir / name;
    hash = (hash ^ artifact_text_checksum(name)[0]) * 1099511628211ull;
    if (!std::filesystem::is_regular_file(path)) continue;
    auto text = file_digest(path);
    for (char ch : text) {
      hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
    }
    auto size = std::filesystem::file_size(path);
    for (int i = 0; i < 8; ++i) {
      hash = (hash ^ ((size >> (i * 8)) & 0xffu)) * 1099511628211ull;
    }
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}
std::string manifest_checksum(const std::filesystem::path& dir) {
  auto manifest = read_text(dir / "manifest.json");
  auto checksum = json_first_string(manifest, "weights_checksum");
  return checksum.empty() ? file_digest(dir / "weights.lkjw") : checksum;
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
  bool accepted_attention = report.attention_backend == "cuda_causal_gqa_bf16_reference" || report.attention_backend == "cudnn_sdpa";
  bool accepted_decoder = kind == "decoder" && impl == "accepted" && accepted_attention && report.decoder_cuda_path && report.decoder_cuda_slice == "full_decoder" && report.decoder_block_backend == "cuda_full_decoder" && report.forward_backend == "cuda_full_decoder" && report.backward_backend == "cuda_full_decoder" && report.decoder_backward_backend == "cuda_full_decoder" && report.embedding_tying == "tok_embeddings:lm_head" && report.kv_cache_backend == kDecoderAcceptedKvCacheBackend && report.decode_backend == kDecoderAcceptedDecodeBackend;
  double tokens_per_second = report.elapsed_seconds > 0.0
      ? static_cast<double>(report.input_tokens) / report.elapsed_seconds : 0.0;
  std::vector<std::string> limitations;
  if (report.run_purpose == "bounded_compatibility_start_check") limitations.push_back("bounded_compatibility_start_check");
  if (!accepted_decoder) {
    limitations.push_back("experimental_not_accepted_cuda_training");
    limitations.push_back(report.decoder_cuda_path ? "partial_cuda_decoder_slice" : "host_reference_forward");
    limitations.push_back(report.decoder_cuda_path ? "decoder_forward_partial" : "host_surrogate_backward");
    if (report.forward_backend != "cuda_full_decoder") limitations.push_back("full_forward_not_accepted"); if (report.backward_backend != "cuda_full_decoder") limitations.push_back("full_backward_not_accepted");
    if (report.attention_backend == "not_implemented") limitations.push_back("attention_not_implemented"); if (report.decoder_backward_backend == "not_implemented") limitations.push_back("decoder_backward_not_implemented");
    if (report.kv_cache_backend == "none") limitations.push_back("kv_cache_not_implemented");
    if (!report.decode_supported) limitations.push_back("autoregressive_decode_unsupported");
  }
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
       << ",\"rmsnorm_backend\":\"" << json_escape(report.rmsnorm_backend)
       << "\",\"rope_backend\":\"" << json_escape(report.rope_backend)
       << "\",\"qkv_projection_backend\":\"" << json_escape(report.qkv_projection_backend) << "\""
       << ",\"attention_backend\":\"" << json_escape(report.attention_backend) << "\""
       << ",\"mlp_backend\":\"" << json_escape(report.mlp_backend)
       << "\",\"decoder_backward_backend\":\"" << json_escape(report.decoder_backward_backend) << "\""
       << ",\"matmul_backend\":\"" << json_escape(report.matmul_backend)
       << "\",\"kv_cache_backend\":\"" << json_escape(report.kv_cache_backend)
       << "\",\"decode_backend\":\"" << json_escape(report.decode_backend)
       << "\",\"cublaslt_workspace_bytes\":" << static_cast<unsigned long long>(report.cublaslt_workspace_bytes)
       << ",\"workspace_high_water_bytes\":" << static_cast<unsigned long long>(report.workspace_high_water_bytes)
       << ",\"workspace_reallocations\":" << report.workspace_reallocations
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
       << ",\"config_path\":\"" << json_escape(report.config_path.string()) << "\",\"config_digest\":\"" << file_digest(report.config_path) << "\""
       << ",\"dataset_path\":\"" << json_escape(report.packed_cache.string())
       << "\",\"packed_cache_path\":\"" << json_escape(report.packed_cache.string()) << "\""
       << ",\"dataset_digest\":\"" << packed_cache_digest(report.packed_cache)
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
       << ",\"loss\":" << report.loss << ",\"loss_finite\":true"
       << ",\"weight_changed\":" << (report.trainable_weight_changed ? "true" : "false")
       << ",\"non_embedding_weight_changed\":" << (report.non_embedding_weight_changed ? "true" : "false")
       << ",\"elapsed_ms\":" << report.elapsed_seconds * 1000.0
       << ",\"elapsed_seconds\":" << report.elapsed_seconds
       << ",\"tokens_per_second\":" << tokens_per_second
       << ",\"checkpoint_path\":\""
       << json_escape(report.checkpoint_dir.string()) << "\""
       << ",\"checkpoint_checksum\":\""
       << manifest_checksum(report.checkpoint_dir) << "\""
       << ",\"export_path\":\"" << json_escape(report.export_dir.string())
       << "\",\"export_checksum\":\"" << manifest_checksum(report.export_dir)
       << "\",\"served_path\":\"" << json_escape(report.served_dir.string())
       << "\",\"logits_checksum\":\""
       << json_escape(report.logits_check_checksum) << "\""
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
std::string transformer_train_report_json(const TransformerTrainReport& report, const CudaStatus& cuda, const std::string& trainer_mode, const std::string& status, const std::string& failure_reason) {
  std::ostringstream out; append_transformer(&out, report, cuda, trainer_mode, status, failure_reason); out << "}"; return out.str(); }
}  // namespace lkjai
