#include "train_real.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#include "cuda_probe.hpp"
#include "capability_json.hpp"
#include "dense_train.hpp"
#include "env.hpp"
#include "json_min.hpp"
#include "training_config.hpp"
#include "train_report.hpp"

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
  auto train_config = env_string("TRAIN_CONFIG", "");
  if (!train_config.empty() &&
      !apply_training_config(train_config, opt, error)) return false;
  if (train_config.empty() &&
      std::filesystem::is_regular_file("configs/training/scratch_40m_12h.json")) {
    train_config = "configs/training/scratch_40m_12h.json";
    if (!apply_training_config(train_config, opt, error)) return false;
  }
  opt->out_dir = env_string("DATA_DIR", opt->out_dir.empty()
                                            ? "/app/data/train"
                                            : opt->out_dir.string());
  opt->model_name = env_string("MODEL_NAME", opt->model_name);
  opt->config_path = env_string("TRAIN_NATIVE_CONFIG", opt->config_path.string());
  opt->packed_cache = env_string(
      "TRAIN_PACKED_CACHE_DIR",
      opt->packed_cache.empty()
          ? opt->out_dir.string() + "/datasets/packed/train-causal_lm_full-seq1024"
          : opt->packed_cache.string());
  opt->max_steps = env_int("TRAIN_MAX_OPTIMIZER_STEPS",
                           env_int("TRAIN_MAX_STEPS", opt->max_steps));
  opt->checkpoint_interval =
      env_int("TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS",
              opt->checkpoint_interval);
  opt->batch_size = env_int("TRAIN_BATCH_SIZE", opt->batch_size);
  opt->seq_len = env_int("TRAIN_SEQUENCE_LEN", opt->seq_len);
  opt->grad_accum = env_int("TRAIN_GRADIENT_ACCUMULATION", opt->grad_accum);
  opt->warmup_steps = env_int("TRAIN_WARMUP_STEPS", opt->warmup_steps);
  opt->seed = env_int("TRAIN_SEED", opt->seed);
  opt->lr = env_float("TRAIN_LEARNING_RATE", opt->lr);
  opt->config_path = value(argc, argv, "--config", opt->config_path.string());
  opt->packed_cache = value(argc, argv, "--packed-cache",
                            opt->packed_cache.string());
  opt->out_dir = value(argc, argv, "--out", opt->out_dir.string());
  opt->batch_size = int_value(argc, argv, "--batch-size", opt->batch_size);
  opt->seq_len = int_value(argc, argv, "--seq-len", opt->seq_len);
  opt->grad_accum = int_value(argc, argv, "--grad-accum", opt->grad_accum);
  opt->max_steps = int_value(argc, argv, "--max-steps", opt->max_steps);
  opt->warmup_steps = int_value(argc, argv, "--warmup-steps", opt->warmup_steps);
  opt->checkpoint_interval =
      int_value(argc, argv, "--checkpoint-interval", opt->checkpoint_interval);
  opt->lr = float_value(argc, argv, "--lr", opt->lr);
  opt->resume_dir = value(argc, argv, "--resume", "");
  opt->export_artifact = value(argc, argv, "--export-artifact", "");
  return true;
}

}  // namespace

int run_corpus_training(int argc, char** argv) {
  if (flag(argc, argv, "--help")) {
    std::cout << "usage: lkjai-native-train --train --packed-cache DIR "
                 "--config FILE --out DIR [--max-steps N]\n";
    return 0;
  }
  DenseTrainOptions opt;
  DenseTrainReport report;
  std::string error;
  if (!options(argc, argv, &opt, &error)) {
    std::cerr << "native training config failed: " << error << "\n";
    return 2;
  }
  if (!run_dense_training(opt, &report, &error)) {
    std::cerr << "native dense CUDA training failed: " << error << "\n";
    return 2;
  }
  auto cuda = cuda_status();
  if (!write_dense_train_report(report, cuda, "train", "pass", "", &error)) {
    std::cerr << error << "\n";
    return 2;
  }
  std::cout << dense_train_report_json(report, cuda, "train", "pass", "")
            << "\n";
  return report.weight_changed ? 0 : 3;
}

}  // namespace lkjai
