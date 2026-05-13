#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lkjai {

struct DenseConfig {
  std::string model = "dense-debug-bf16";
  int vocab_size = 8192;
  int context = 1024;
  int layers = 1;
  int hidden_size = 32;
  int heads = 4;
  int kv_heads = 1;
  int head_dim = 8;
  int ffn_size = 64;
  int seed = 1337;
};

struct DenseTrainOptions {
  std::filesystem::path packed_cache;
  std::filesystem::path config_path = "configs/native/native_debug_bf16.json";
  std::filesystem::path out_dir;
  std::filesystem::path resume_dir;
  std::filesystem::path export_artifact;
  std::filesystem::path tokenizer_path;
  std::string model_name = "decoder-40m-3070";
  std::string model_kind = "dense";
  std::string run_purpose;
  int batch_size = 1;
  int seq_len = 0;
  int grad_accum = 1;
  int max_steps = 2;
  int target_seconds = 0;
  int warmup_steps = 0;
  int checkpoint_interval = 1;
  int loss_sample_interval = 0;
  int seed = -1;
  float lr = 1.0e-3f;
  std::string lr_schedule = "warmup_constant";
  float min_lr_fraction = 0.0f;
  std::filesystem::path train_config_path;
};

struct DenseLossSample {
  int step = 0;
  double loss = 0.0;
};

struct DenseWeightTensorDelta {
  std::string name;
  uint64_t checked_elements = 0;
  uint64_t changed_elements = 0;
  double max_abs_delta = 0.0;
  double mean_abs_delta = 0.0;
  std::string status = "fail";
};

struct DenseWeightChangeReport {
  int checked_tensors = 0;
  int changed_tensors = 0;
  uint64_t checked_elements = 0;
  uint64_t changed_elements = 0;
  double max_abs_delta = 0.0;
  double mean_abs_delta = 0.0;
  double tolerance = 0.0;
  std::vector<DenseWeightTensorDelta> tensors;
  std::string status = "unknown";
  std::string reason;
};

struct DenseTrainReport {
  int steps = 0;
  int start_step = 0;
  int microsteps = 0;
  int input_tokens = 0;
  int loss_tokens = 0;
  int batch_size = 0;
  int seq_len = 0;
  int grad_accum = 1;
  int target_seconds = 0;
  bool deadline_hit = false;
  std::string stop_reason = "max_steps";
  std::string lr_schedule = "warmup_constant";
  double learning_rate = 0.0;
  double min_learning_rate_fraction = 0.0;
  double final_learning_rate = 0.0;
  double initial_loss = 0.0;
  double loss = 0.0;
  int loss_sample_interval = 0;
  std::vector<DenseLossSample> loss_samples;
  double best_loss = 0.0;
  int best_loss_step = 0;
  double loss_delta = 0.0;
  double loss_decrease_fraction = 0.0;
  double first_quarter_loss_mean = 0.0;
  double last_quarter_loss_mean = 0.0;
  std::string learning_status = "unknown";
  bool weight_changed = false;
  DenseWeightChangeReport weight_change;
  std::string logits_checksum;
  bool logits_check_passed = false;
  std::string logits_check_json;
  std::string logits_check_checksum;
  std::string failure_reason;
  std::string run_purpose;
  std::filesystem::path train_config_path;
  std::filesystem::path config_path;
  std::filesystem::path packed_cache;
  std::filesystem::path checkpoint_dir;
  std::filesystem::path best_checkpoint_dir;
  std::filesystem::path export_dir;
  std::filesystem::path served_dir;
  double elapsed_seconds = 0.0;
  double batch_load_seconds = 0.0;
  double h2d_seconds = 0.0;
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double optimizer_seconds = 0.0;
  double checkpoint_seconds = 0.0;
  double export_seconds = 0.0;
  uint64_t dense_step_logits_bytes = 0;
  uint64_t dense_step_grad_logits_bytes = 0;
  uint64_t dense_step_d_hidden_bytes = 0;
  uint64_t dense_logits_readback_bytes = 0;
  uint64_t cublaslt_workspace_bytes = 0;
  bool dense_autotune_enabled = false;
  std::string dense_autotune_mode;
  std::string dense_workspace_sweep_bytes;
  int dense_cublaslt_logits_algo_id = -1;
  int dense_cublaslt_head_grad_algo_id = -1;
  int dense_cublaslt_hidden_grad_algo_id = -1;
  uint64_t dense_cublaslt_logits_workspace_bytes = 0;
  uint64_t dense_cublaslt_head_grad_workspace_bytes = 0;
  uint64_t dense_cublaslt_hidden_grad_workspace_bytes = 0;
  std::string dense_allocator_backend;
  bool dense_async_alloc_supported = false;
  uint64_t dense_mempool_release_threshold_bytes = 0;
  uint64_t dense_workspace_high_water_bytes = 0;
  int dense_workspace_reallocations = 0;
  std::string dense_timing_mode = "deferred";
  bool dense_head_f32_cache_enabled = true;
  int dense_head_f32_cache_refreshes = 0;
  int dense_stream_count = 2;
  int dense_batch_slot_count = 3;
  bool copy_compute_overlap_enabled = false;
  std::string batch_staging_backend = "triple_slot_pinned_direct_read";
};

bool load_dense_config(const std::filesystem::path& path, DenseConfig* config,
                       std::string* error);
bool run_dense_training(const DenseTrainOptions& opt, DenseTrainReport* report,
                        std::string* error);

}  // namespace lkjai
