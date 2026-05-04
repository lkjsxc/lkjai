#include "train_real.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include "cuda_probe.hpp"
#include "dense_train.hpp"
#include "env.hpp"
#include "training_config.hpp"
#include "train_report.hpp"
#include "transformer_train.hpp"
namespace lkjai {
namespace {
bool flag(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) return true;
  }
  return false;
}
std::string value(int argc, char** argv, const std::string& name,
                  const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return fallback;
}
int int_value(int argc, char** argv, const std::string& name, int fallback) {
  try {
    return std::stoi(value(argc, argv, name, std::to_string(fallback)));
  } catch (...) {
    return fallback;
  }
}
float float_value(int argc, char** argv, const std::string& name,
                  float fallback) {
  try {
    return std::stof(value(argc, argv, name, std::to_string(fallback)));
  } catch (...) {
    return fallback;
  }
}
float env_float(const char* name, float fallback) {
  try {
    return std::stof(env_string(name, std::to_string(fallback)));
  } catch (...) {
    return fallback;
  }
}

bool options(int argc, char** argv, DenseTrainOptions* opt,
             std::string* error) {
  bool config_explicit = false;
  auto train_config = env_string("TRAIN_CONFIG", "");
  if (!train_config.empty() &&
      !apply_training_config(train_config, opt, error)) return false;
  if (train_config.empty() &&
      std::filesystem::is_regular_file("configs/training/scratch_40m_12h.json")) {
    train_config = "configs/training/scratch_40m_12h.json";
    if (!apply_training_config(train_config, opt, error)) return false;
  }
  if (opt->config_path != std::filesystem::path("configs/native/native_debug_bf16.json")) {
    config_explicit = true;
  }
  opt->out_dir = env_string("DATA_DIR", opt->out_dir.empty()
                                            ? "/app/data/train"
                                            : opt->out_dir.string());
  opt->model_name = env_string("MODEL_NAME", opt->model_name);
  auto env_config = env_string("TRAIN_NATIVE_CONFIG", "");
  if (!env_config.empty()) {
    opt->config_path = env_config;
    config_explicit = true;
  }
  opt->packed_cache = env_string(
      "TRAIN_PACKED_CACHE_DIR",
      opt->packed_cache.empty()
          ? opt->out_dir.string() + "/datasets/packed/train-causal_lm_full-seq1024"
          : opt->packed_cache.string());
  opt->max_steps = env_int("TRAIN_MAX_OPTIMIZER_STEPS",
                           env_int("TRAIN_MAX_STEPS", opt->max_steps));
  opt->target_seconds = env_int("TRAIN_TARGET_SECONDS", opt->target_seconds);
  opt->checkpoint_interval = env_int("TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS", opt->checkpoint_interval);
  opt->loss_sample_interval =
      env_int("TRAIN_LOSS_SAMPLE_INTERVAL", opt->loss_sample_interval);
  opt->batch_size = env_int("TRAIN_BATCH_SIZE", opt->batch_size);
  opt->seq_len = env_int("TRAIN_SEQUENCE_LEN", opt->seq_len);
  opt->grad_accum = env_int("TRAIN_GRADIENT_ACCUMULATION", opt->grad_accum);
  opt->warmup_steps = env_int("TRAIN_WARMUP_STEPS", opt->warmup_steps);
  opt->seed = env_int("TRAIN_SEED", opt->seed);
  opt->lr = env_float("TRAIN_LEARNING_RATE", opt->lr);
  auto env_kind = env_string("TRAIN_MODEL_KIND", "");
  if (!env_kind.empty()) opt->model_kind = env_kind;
  opt->run_purpose = env_string("TRAIN_RUN_PURPOSE", opt->run_purpose);
  auto cli_config = value(argc, argv, "--config", "");
  if (!cli_config.empty()) {
    opt->config_path = cli_config;
    config_explicit = true;
  }
  opt->model_kind = value(argc, argv, "--mode", opt->model_kind);
  if (opt->model_kind != "dense" && opt->model_kind != "transformer" &&
      opt->model_kind != "decoder") {
    *error = "model kind must be dense, transformer, or decoder";
    return false;
  }
  if (!config_explicit &&
      opt->config_path == std::filesystem::path("configs/native/native_debug_bf16.json")) {
    if (opt->model_kind == "transformer")
      opt->config_path = "configs/native/native_transformer_debug_bf16.json";
    else if (opt->model_kind == "decoder")
      opt->config_path = "configs/native/decoder_debug_bf16.json";
  }
  opt->packed_cache = value(argc, argv, "--packed-cache",
                            opt->packed_cache.string());
  opt->out_dir = value(argc, argv, "--out", opt->out_dir.string());
  opt->batch_size = int_value(argc, argv, "--batch-size", opt->batch_size);
  opt->seq_len = int_value(argc, argv, "--seq-len", opt->seq_len);
  opt->grad_accum = int_value(argc, argv, "--grad-accum", opt->grad_accum);
  opt->max_steps = int_value(argc, argv, "--max-steps", opt->max_steps);
  opt->target_seconds = int_value(argc, argv, "--target-seconds", opt->target_seconds);
  opt->warmup_steps = int_value(argc, argv, "--warmup-steps", opt->warmup_steps);
  opt->checkpoint_interval =
      int_value(argc, argv, "--checkpoint-interval", opt->checkpoint_interval);
  opt->loss_sample_interval = int_value(
      argc, argv, "--loss-sample-interval", opt->loss_sample_interval);
  opt->lr = float_value(argc, argv, "--lr", opt->lr);
  opt->resume_dir = value(argc, argv, "--resume", "");
  opt->export_artifact = value(argc, argv, "--export-artifact", "");
  opt->run_purpose = value(argc, argv, "--run-purpose", opt->run_purpose);
  if (opt->run_purpose.empty()) opt->run_purpose = "accepted_training";
  return true;
}

TransformerTrainOptions transformer_options(const DenseTrainOptions& in) {
  TransformerTrainOptions out;
  out.packed_cache = in.packed_cache;
  out.config_path = in.config_path;
  out.out_dir = in.out_dir;
  out.resume_dir = in.resume_dir;
  out.export_artifact = in.export_artifact;
  out.model_name = in.model_name;
  out.model_kind = "transformer";
  out.run_purpose = in.run_purpose;
  out.batch_size = in.batch_size;
  out.seq_len = in.seq_len;
  out.grad_accum = in.grad_accum;
  out.max_steps = in.max_steps;
  out.target_seconds = in.target_seconds;
  out.warmup_steps = in.warmup_steps;
  out.checkpoint_interval = in.checkpoint_interval;
  out.seed = in.seed;
  out.lr = in.lr;
  out.train_config_path = in.train_config_path;
  return out;
}

}  // namespace

int run_corpus_training(int argc, char** argv) {
  if (flag(argc, argv, "--help")) {
    std::cout << "usage: lkjai-native-train --train [--mode dense|transformer|decoder]\n";
    return 0;
  }
  DenseTrainOptions opt;
  DenseTrainReport report;
  std::string error;
  if (!options(argc, argv, &opt, &error)) {
    std::cerr << "native training config failed: " << error << "\n";
    return 2;
  }
  if (opt.model_kind == "transformer" || opt.model_kind == "decoder") {
    TransformerTrainReport transformer_report;
    auto transformer_opt = transformer_options(opt);
    transformer_opt.model_kind = opt.model_kind;
    if (!run_transformer_training(transformer_opt, &transformer_report, &error)) {
      std::cerr << "native transformer CUDA training failed: " << error << "\n";
      return 2;
    }
    auto cuda = cuda_status();
    if (!write_transformer_train_report(transformer_report, cuda, "train",
                                        "success", "", &error)) {
      std::cerr << error << "\n";
      return 2;
    }
    std::cout << transformer_train_report_json(transformer_report, cuda, "train",
                                               "success", "")
              << "\n";
    return transformer_report.trainable_weight_changed ? 0 : 3;
  }
  if (!run_dense_training(opt, &report, &error)) {
    std::cerr << "native dense CUDA training failed: " << error << "\n";
    return 2;
  }
  auto cuda = cuda_status();
  if (!write_dense_train_report(report, cuda, "train", "success", "", &error)) {
    std::cerr << error << "\n";
    return 2;
  }
  std::cout << dense_train_report_json(report, cuda, "train", "success", "")
            << "\n";
  return report.weight_changed ? 0 : 3;
}
}  // namespace lkjai
