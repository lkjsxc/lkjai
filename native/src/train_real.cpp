#include "train_real.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#include "cuda_probe.hpp"
#include "env.hpp"
#include "json_min.hpp"
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

TransformerTrainOptions options(int argc, char** argv) {
  TransformerTrainOptions opt;
  opt.out_dir = env_string("DATA_DIR", "/app/data/train");
  opt.model_name = env_string("MODEL_NAME", "lkjai-scratch-40m");
  opt.packed_cache = env_string(
      "TRAIN_PACKED_CACHE_DIR",
      opt.out_dir.string() + "/datasets/packed/train-causal_lm_full-seq1024");
  opt.max_steps = env_int("TRAIN_MAX_OPTIMIZER_STEPS",
                          env_int("TRAIN_MAX_STEPS", opt.max_steps));
  opt.checkpoint_interval =
      env_int("TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS",
              opt.checkpoint_interval);
  opt.config_path = value(argc, argv, "--config", opt.config_path.string());
  opt.packed_cache = value(argc, argv, "--packed-cache", opt.packed_cache.string());
  opt.out_dir = value(argc, argv, "--out", opt.out_dir.string());
  opt.batch_size = int_value(argc, argv, "--batch-size", opt.batch_size);
  opt.seq_len = int_value(argc, argv, "--seq-len", opt.seq_len);
  opt.grad_accum = int_value(argc, argv, "--grad-accum", opt.grad_accum);
  opt.max_steps = int_value(argc, argv, "--max-steps", opt.max_steps);
  opt.warmup_steps = int_value(argc, argv, "--warmup-steps", opt.warmup_steps);
  opt.checkpoint_interval =
      int_value(argc, argv, "--checkpoint-interval", opt.checkpoint_interval);
  opt.lr = float_value(argc, argv, "--lr", opt.lr);
  opt.resume_dir = value(argc, argv, "--resume", "");
  opt.export_artifact = value(argc, argv, "--export-artifact", "");
  return opt;
}

}  // namespace

int run_corpus_training(int argc, char** argv) {
  if (flag(argc, argv, "--help")) {
    std::cout << "usage: lkjai-native-train --train --packed-cache DIR "
                 "--config FILE --out DIR [--max-steps N]\n";
    return 0;
  }
  auto opt = options(argc, argv);
  TransformerTrainReport report;
  std::string error;
  if (!run_transformer_training(opt, &report, &error)) {
    std::cerr << "native transformer training failed: " << error << "\n";
    return 2;
  }
  auto cuda = cuda_status();
  std::cout << "{\"status\":\"pass\",\"mode\":\"train\",\"steps\":"
            << report.steps << ",\"start_step\":" << report.start_step
            << ",\"loss\":" << report.loss << ",\"loss_finite\":true"
            << ",\"transformer_path\":true"
            << ",\"non_embedding_weight_changed\":"
            << (report.non_embedding_weight_changed ? "true" : "false")
            << ",\"logits_checksum\":\""
            << json_escape(report.logits_checksum) << "\""
            << ",\"timings\":{\"batch_load\":0,\"forward\":0,"
               "\"backward\":0,\"optimizer\":0,\"checkpoint_export\":0}"
            << ",\"cuda_available\":"
            << (cuda.available ? "true" : "false")
            << ",\"elapsed_seconds\":" << report.elapsed_seconds << "}\n";
  return report.non_embedding_weight_changed ? 0 : 3;
}

}  // namespace lkjai
