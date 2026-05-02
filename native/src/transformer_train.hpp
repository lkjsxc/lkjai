#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

struct TransformerConfig {
  std::string model = "native-debug-bf16";
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
  std::string model_name = "lkjai-scratch-40m";
  std::string model_kind = "transformer";
  int batch_size = 1;
  int seq_len = 0;
  int grad_accum = 1;
  int max_steps = 2;
  int warmup_steps = 0;
  int checkpoint_interval = 1;
  int seed = -1;
  float lr = 1.0e-3f;
  std::filesystem::path train_config_path;
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
  double initial_loss = 0.0;
  double loss = 0.0;
  bool non_embedding_weight_changed = false;
  bool logits_check_passed = false;
  std::string logits_check_json;
  std::string logits_check_checksum;
  std::string logits_checksum;
  std::filesystem::path train_config_path;
  std::filesystem::path config_path;
  std::filesystem::path packed_cache;
  std::filesystem::path checkpoint_dir;
  std::filesystem::path export_dir;
  std::filesystem::path served_dir;
  double elapsed_seconds = 0.0;
  double batch_load_seconds = 0.0;
  double h2d_seconds = 0.0;
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double optimizer_seconds = 0.0;
  double checkpoint_export_seconds = 0.0;
  double export_seconds = 0.0;
};

bool load_transformer_config(const std::filesystem::path& path,
                             TransformerConfig* config, std::string* error);
bool run_transformer_training(const TransformerTrainOptions& opt,
                              TransformerTrainReport* report,
                              std::string* error);
bool transformer_logits_check(const std::filesystem::path& model_dir,
                              const std::string& token_csv,
                              std::string* json, std::string* error);

}  // namespace lkjai
