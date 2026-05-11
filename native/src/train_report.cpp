#include "train_report.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "capability_json.hpp"
#include "dense_report_util.hpp"
#include "dense_weight_change.hpp"
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
void append_common(std::ostringstream* out, const DenseTrainReport& report,
                   const CudaStatus& cuda, const std::string& trainer_mode,
                   const std::string& status,
                   const std::string& failure_reason) {
  double elapsed_ms = report.elapsed_seconds * 1000.0;
  double tokens_per_second = report.elapsed_seconds > 0.0
      ? static_cast<double>(report.input_tokens) / report.elapsed_seconds : 0.0;
  *out << "{\"schema\":\"lkjai-train-report\""
       << ",\"trainer_mode\":\"" << json_escape(trainer_mode) << "\""
       << ",\"mode\":\"" << json_escape(trainer_mode) << "\""
       << ",\"run_purpose\":\"" << json_escape(report.run_purpose) << "\""
       << ",\"model_kind\":\"dense\""
       << ",\"accepted_cuda_training\":true"
       << ",\"implementation_status\":\"accepted\""
       << ",\"forward_backend\":\"cuda_bf16_cublaslt\""
       << ",\"backward_backend\":\"cuda_bf16_cublaslt_scatter\""
       << ",\"backward_gemm_enabled\":true"
       << ",\"embedding_grad_backend\":\"token_scatter_add_fp32\""
       << ",\"loss_kernel_backend\":\"block_row_softmax_fp32\""
       << ",\"loss_readback_mode\":\"optimizer_step_deferred_pinned\""
       << ",\"logits_readback_mode\":\"single_row_capture\""
       << ",\"dense_stream_count\":" << report.dense_stream_count
       << ",\"dense_batch_slot_count\":" << report.dense_batch_slot_count
       << ",\"copy_compute_overlap_enabled\":"
       << (report.copy_compute_overlap_enabled ? "true" : "false")
       << ",\"batch_staging_backend\":\"" << json_escape(report.batch_staging_backend) << "\""
       << ",\"optimizer_backend\":\"cuda_adamw_fp32\""
       << ",\"cuda_probe_passed\":"
       << (cuda_required_ok(cuda) ? "true" : "false")
       << ",\"status\":\"" << json_escape(status) << "\""
       << ",\"failure_reason\":\"" << json_escape(failure_reason) << "\""
       << ",\"limitations\":["
       << (report.run_purpose == "bounded_diagnostic_start_check"
               ? "\"bounded_diagnostic_start_check\","
               : "")
       << "\"single_gpu_only\","
          "\"dense_embedding_lm_head_only\","
          "\"autoregressive_decode_unsupported\"]"
       << ",\"precision_mode\":\"fp32-master-bf16-shadow-bf16-export\""
       << ",\"master_dtype\":\"f32\""
       << ",\"shadow_dtype\":\"bf16\""
       << ",\"accumulation_dtype\":\"f32\""
       << ",\"export_dtype\":\"bf16\""
       << ",\"dense_cuda_path\":true"
       << ",\"loader_backend\":\"persistent_packed_cache_reader\",\"row_layout\":\"dense_physical_bxseq_masked_final_token\",\"matmul_plan_cache_enabled\":true,\"buffer_reuse_enabled\":true,\"timing_source\":\""
       << (report.dense_timing_mode == "deferred" ? "cuda_events_deferred_slot_sync" : "cuda_events_with_boundary_sync")
       << "\""
       << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
       << ",\"cuda_device_name\":\"" << json_escape(cuda.device) << "\""
       << ",\"cuda_driver_version\":" << cuda.cuda_driver_version
       << ",\"cuda_runtime_version\":" << cuda.cuda_runtime_version
       << ",\"cudnn_version\":" << cuda.cudnn_version
       << ",\"cuda_device_count\":" << cuda.device_count
       << ",\"cuda_device_index\":" << cuda.device_index
       << ",\"cuda_total_global_memory\":"
       << static_cast<unsigned long long>(cuda.total_global_memory)
       << ",\"cuda_sm_count\":" << cuda.sm_count
       << ",\"cuda_arch_flags\":\"" << json_escape(LKJAI_CUDA_ARCH_FLAGS)
       << "\""
       << ",\"git_commit\":\"" << json_escape(LKJAI_GIT_COMMIT) << "\""
       << ",\"build_type\":\"" << json_escape(LKJAI_BUILD_TYPE) << "\""
       << ",\"config_path\":\"" << json_escape(report.config_path.string())
       << "\""
       << ",\"config_digest\":\""
       << train_report_file_digest(report.config_path) << "\""
       << ",\"dataset_path\":\"" << json_escape(report.packed_cache.string())
       << "\""
       << ",\"packed_cache_path\":\""
       << json_escape(report.packed_cache.string()) << "\""
       << ",\"dataset_digest\":\""
       << train_report_packed_cache_digest(report.packed_cache)
       << "\""
       << ",\"train_config_path\":\""
       << json_escape(report.train_config_path.string()) << "\""
       << ",\"seed\":" << json_int_value(read_text(report.config_path), "seed", 0)
       << ",\"batch_size\":" << report.batch_size
       << ",\"seq_len\":" << report.seq_len
       << ",\"grad_accum\":" << report.grad_accum;
  append_dense_run_control_fields(out, report);
  *out << ",\"parameter_count\":" << dense_report_parameter_count(report)
       << ",\"dense_step_logits_bytes\":"
       << static_cast<unsigned long long>(report.dense_step_logits_bytes)
       << ",\"dense_step_grad_logits_bytes\":"
       << static_cast<unsigned long long>(report.dense_step_grad_logits_bytes)
       << ",\"dense_step_d_hidden_bytes\":"
       << static_cast<unsigned long long>(report.dense_step_d_hidden_bytes)
       << ",\"dense_logits_readback_bytes\":"
       << static_cast<unsigned long long>(report.dense_logits_readback_bytes)
       << ",\"cublaslt_workspace_bytes\":"
       << static_cast<unsigned long long>(report.cublaslt_workspace_bytes);
  append_dense_tuning_fields(out, report);
  *out
       << ",\"tokens_seen\":" << report.input_tokens
       << ",\"input_tokens\":" << report.input_tokens
       << ",\"loss_tokens\":" << report.loss_tokens
       << ",\"optimizer_steps\":" << report.steps
       << ",\"steps\":" << report.steps
       << ",\"start_step\":" << report.start_step
       << ",\"microsteps\":" << report.microsteps
       << ",\"initial_loss\":" << report.initial_loss
       << ",\"loss\":" << report.loss
       << ",\"loss_samples\":";
  append_dense_loss_samples(out, report.loss_samples);
  *out << ",\"loss_sample_interval\":" << report.loss_sample_interval
       << ",\"best_loss\":" << report.best_loss
       << ",\"best_loss_step\":" << report.best_loss_step
       << ",\"loss_delta\":" << report.loss_delta
       << ",\"loss_decrease_fraction\":" << report.loss_decrease_fraction
       << ",\"first_quarter_loss_mean\":"
       << report.first_quarter_loss_mean
       << ",\"last_quarter_loss_mean\":" << report.last_quarter_loss_mean
       << ",\"learning_status\":\""
       << json_escape(report.learning_status) << "\""
       << ",\"loss_finite\":" << (std::isfinite(report.loss) ? "true" : "false")
       << ",\"weight_changed\":"
       << (report.weight_changed ? "true" : "false") << ",\"weight_change\":";
  append_dense_weight_change_json(*out, report.weight_change);
  *out << ",\"elapsed_ms\":" << elapsed_ms
       << ",\"elapsed_seconds\":" << report.elapsed_seconds
       << ",\"tokens_per_second\":" << tokens_per_second
       << ",\"checkpoint_path\":\""
       << json_escape(report.checkpoint_dir.string()) << "\""
       << ",\"checkpoint_checksum\":\""
       << train_report_manifest_checksum(report.checkpoint_dir) << "\""
       << ",\"export_path\":\"" << json_escape(report.export_dir.string())
       << "\""
       << ",\"export_checksum\":\""
       << train_report_manifest_checksum(report.export_dir)
       << "\""
       << ",\"served_path\":\"" << json_escape(report.served_dir.string())
       << "\""
       << ",\"logits_checksum\":\""
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
       << ",\"checkpoint\":" << report.checkpoint_seconds
       << ",\"export\":" << report.export_seconds << "}"
       << ",\"capability\":{" << capability_json_fields(cuda) << "}";
}
}  // namespace
std::string dense_train_report_json(const DenseTrainReport& report,
                                    const CudaStatus& cuda,
                                    const std::string& trainer_mode,
                                    const std::string& status,
                                    const std::string& failure_reason) {
  std::ostringstream out;
  append_common(&out, report, cuda, trainer_mode, status, failure_reason);
  out << "}";
  return out.str();
}
bool write_dense_train_report(const DenseTrainReport& report,
                              const CudaStatus& cuda,
                              const std::string& trainer_mode,
                              const std::string& status,
                              const std::string& failure_reason,
                              std::string* error) {
  auto path = report.checkpoint_dir.parent_path().parent_path() / "runs" /
              "train-report.json";
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    *error = "failed to write train report: " + path.string();
    return false;
  }
  out << dense_train_report_json(report, cuda, trainer_mode, status,
                                 failure_reason)
      << "\n";
  return true;
}
}  // namespace lkjai
