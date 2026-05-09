#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace lkjai {

struct TransformerConfig {
  std::string model = "native-debug-bf16";
  std::string kind = "transformer";
  std::string dtype = "bf16";
  int vocab_size = 256;
  int context = 16;
  int layers = 1;
  int hidden_size = 32;
  int heads = 4;
  int kv_heads = 4;
  int head_dim = 8;
  int ffn_size = 64;
  std::string activation = "swiglu";
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1.0e-5f;
  bool tie_embeddings = false;
  int seed = 1337;
};

struct TransformerTrainOptions {
  std::filesystem::path packed_cache;
  std::filesystem::path config_path = "configs/native/native_debug_bf16.json";
  std::filesystem::path out_dir;
  std::filesystem::path resume_dir;
  std::filesystem::path export_artifact;
  std::filesystem::path tokenizer_path;
  std::string model_name = "lkjai-scratch-40m";
  std::string model_kind = "transformer";
  std::string run_purpose;
  int batch_size = 1;
  int seq_len = 0;
  int grad_accum = 1;
  int max_steps = 2;
  int target_seconds = 0;
  int warmup_steps = 0;
  int checkpoint_interval = 1;
  int seed = -1;
  float lr = 1.0e-3f;
  std::filesystem::path train_config_path;
};

struct DecoderForwardProbeReport {
  bool recorded = false;
  std::string status = "not_run";
  bool rmsnorm = false;
  bool rope = false;
  bool qkv_projection = false;
  bool attention = false;
  bool output_projection = false;
  bool attention_residual = false;
  bool mlp_norm = false;
  bool swiglu = false;
  bool down_projection = false;
  bool block_residual = false;
  bool output_finite = false;
  int batch = 0;
  int sequence = 0;
  int output_rows = 0;
  int output_hidden_size = 0;
  uint64_t workspace_bytes = 0;
};

struct DecoderWeightChangePart {
  double max_abs_delta = 0.0;
  uint64_t changed_elements = 0;
  int changed_tensors = 0;
};

struct DecoderWeightChangeReport {
  DecoderWeightChangePart embedding;
  DecoderWeightChangePart lm_head;
  DecoderWeightChangePart non_embedding;
  DecoderWeightChangePart decoder_block;
  int changed_tensors = 0;
};

struct TransformerTrainReport {
  int steps = 0;
  int start_step = 0;
  int microsteps = 0;
  int input_tokens = 0;
  int loss_tokens = 0;
  int batch_size = 0;
  int seq_len = 0;
  int grad_accum = 1;
  int layers = 0;
  int heads = 0;
  int kv_heads = 0;
  int hidden_size = 0;
  int head_dim = 0;
  int ffn_size = 0;
  int context = 0;
  long long parameter_count = 0;
  int target_seconds = 0;
  bool deadline_hit = false;
  std::string stop_reason = "max_steps";
  double initial_loss = 0.0;
  double loss = 0.0;
  bool embedding_weight_changed = false;
  bool lm_head_weight_changed = false;
  bool non_embedding_weight_changed = false;
  bool decoder_block_weight_changed = false;
  bool trainable_weight_changed = false;
  bool logits_check_passed = false;
  std::string logits_check_json;
  std::string logits_check_checksum;
  std::string logits_checksum;
  std::string run_purpose;
  std::filesystem::path train_config_path;
  std::filesystem::path config_path;
  std::filesystem::path packed_cache;
  std::filesystem::path checkpoint_dir;
  std::filesystem::path export_dir;
  std::filesystem::path served_dir;
  std::string model_kind = "transformer";
  double elapsed_seconds = 0.0;
  double batch_load_seconds = 0.0;
  double h2d_seconds = 0.0;
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double optimizer_seconds = 0.0;
  double checkpoint_export_seconds = 0.0;
  double export_seconds = 0.0;
  bool decoder_cuda_path = false;
  std::string implementation_status = "experimental";
  std::string transformer_status = "experimental";
  std::string decoder_status = "not_applicable";
  std::string forward_backend = "host_reference";
  std::string backward_backend = "host_surrogate";
  std::string optimizer_backend = "host_adamw_fp32";
  std::string attention_backend = "host_reference";
  std::string matmul_backend = "host_reference";
  std::string rmsnorm_backend = "host_reference";
  std::string rope_backend = "host_reference";
  std::string qkv_projection_backend = "host_reference";
  std::string mlp_backend = "host_reference";
  std::string decoder_backward_backend = "not_applicable";
  std::string kv_cache_backend = "none";
  std::string decode_backend = "unsupported";
  std::string decoder_cuda_slice = "none";
  std::string decoder_block_backend = "host_reference";
  bool decoder_block_forward_in_training = false;
  int decoder_block_forward_steps = 0;
  DecoderForwardProbeReport decoder_forward_probe;
  DecoderWeightChangeReport decoder_weight_change;
  bool decode_supported = false;
  std::string embedding_tying = "none";
  int trainable_tensor_count = 0;
  uint64_t cublaslt_workspace_bytes = 0;
  uint64_t workspace_high_water_bytes = 0;
  int workspace_reallocations = 0;
};

bool load_transformer_config(const std::filesystem::path& path,
                             TransformerConfig* config, std::string* error);
bool run_transformer_training(const TransformerTrainOptions& opt,
                              TransformerTrainReport* report,
                              std::string* error);
bool run_decoder_cuda_slice_training(const TransformerTrainOptions& opt,
                                     TransformerTrainReport* report,
                                     std::string* error);
bool transformer_logits_check(const std::filesystem::path& model_dir,
                              const std::string& token_csv,
                              std::string* json, std::string* error);

}  // namespace lkjai
