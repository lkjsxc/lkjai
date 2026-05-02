#pragma once

#include <filesystem>
#include <string>

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
  std::string model_name = "lkjai-scratch-40m";
  int batch_size = 1;
  int seq_len = 0;
  int grad_accum = 1;
  int max_steps = 2;
  int warmup_steps = 0;
  int checkpoint_interval = 1;
  float lr = 1.0e-3f;
};

struct DenseTrainReport {
  int steps = 0;
  int start_step = 0;
  double initial_loss = 0.0;
  double loss = 0.0;
  bool weight_changed = false;
  std::string logits_checksum;
  double elapsed_seconds = 0.0;
  double batch_load_seconds = 0.0;
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double optimizer_seconds = 0.0;
  double checkpoint_seconds = 0.0;
  double export_seconds = 0.0;
};

bool load_dense_config(const std::filesystem::path& path, DenseConfig* config,
                       std::string* error);
bool run_dense_training(const DenseTrainOptions& opt, DenseTrainReport* report,
                        std::string* error);

}  // namespace lkjai
