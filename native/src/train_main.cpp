#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "cuda_probe.hpp"
#include "capability_json.hpp"
#include "dense_train.hpp"
#include "env.hpp"
#include "json_min.hpp"
#include "train_real.hpp"

namespace {

bool has_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) return true;
  }
  return false;
}

int int_arg(int argc, char** argv, const std::string& flag, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) return std::stoi(argv[i + 1]);
  }
  return fallback;
}

void write_u16(std::ofstream& out, uint16_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u64(std::ofstream& out, uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void prepare_smoke_fixture(const std::filesystem::path& root,
                           std::filesystem::path* cache,
                           std::filesystem::path* config) {
  *cache = root / "datasets" / "packed" / "train-causal_lm_full-seq1024";
  std::filesystem::create_directories(*cache);
  std::ofstream(*cache / "metadata.json")
      << "{\"format\":\"lkjai-packed-cache-v2\",\"split\":\"train\","
         "\"objective\":\"causal_lm_full\",\"sequence_len\":16,"
         "\"vocab_size\":256,\"token_dtype\":\"uint16\",\"row_count\":2,"
         "\"token_count\":32}\n";
  std::ofstream tokens(*cache / "tokens.bin", std::ios::binary);
  std::ofstream mask(*cache / "loss_mask.bin", std::ios::binary);
  for (int i = 0; i < 32; ++i) {
    write_u16(tokens, static_cast<uint16_t>((i % 2) + 1));
    mask.put('\1');
  }
  std::ofstream starts(*cache / "starts.bin", std::ios::binary);
  write_u64(starts, 0);
  write_u64(starts, 16);
  *config = root / "native_debug_bf16.json";
  std::ofstream(*config)
      << "{\"model\":\"native-debug-bf16\",\"dtype\":\"bf16\","
         "\"vocab_size\":256,\"context\":16,\"layers\":1,"
         "\"hidden_size\":32,\"heads\":4,\"kv_heads\":4,"
         "\"head_dim\":8,\"ffn_size\":64,\"activation\":\"swiglu\","
         "\"rope_theta\":10000,\"rms_norm_eps\":0.00001,"
         "\"tie_embeddings\":true,\"seed\":1337}\n";
}

lkjai::DenseTrainReport run_smoke_training(int steps) {
  auto data = std::filesystem::path(lkjai::env_string("DATA_DIR", "/tmp/lkjai-native-smoke"));
  auto model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
  std::filesystem::path cache;
  std::filesystem::path config;
  prepare_smoke_fixture(data, &cache, &config);
  lkjai::DenseTrainOptions opt;
  opt.packed_cache = cache;
  opt.config_path = config;
  opt.out_dir = data;
  opt.model_name = model;
  opt.seq_len = 16;
  opt.max_steps = steps;
  lkjai::DenseTrainReport report;
  std::string error;
  if (!lkjai::run_dense_training(opt, &report, &error)) {
    throw std::runtime_error(error);
  }
  return report;
}

}  // namespace

int main(int argc, char** argv) {
  int steps = int_arg(argc, argv, "--steps", 2);
  auto cuda = lkjai::cuda_status();
  if (has_flag(argc, argv, "--help")) {
    std::cout << "usage: lkjai-native-train --smoke [--steps N] | --train "
                 "[--packed-cache DIR] [--config FILE]\n";
    return 0;
  }
  if (has_flag(argc, argv, "--train")) {
    return lkjai::run_corpus_training(argc, argv);
  }
  if (!has_flag(argc, argv, "--smoke")) {
    std::cerr << "native trainer requires --train or --smoke\n";
    return 2;
  }
  auto started = std::chrono::steady_clock::now();
  for (int step = 1; step <= steps; ++step) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::cerr << "{\"event\":\"native_dense_cuda_smoke_step\",\"step\":" << step
              << "}\n";
  }
  auto report = run_smoke_training(steps);
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "{\"status\":\"pass\",\"mode\":\"smoke\",\"steps\":" << steps
            << ",\"optimizer_steps\":" << report.steps
            << ",\"start_step\":" << report.start_step
            << ",\"microsteps\":" << report.microsteps
            << ",\"batch_size\":" << report.batch_size
            << ",\"seq_len\":" << report.seq_len
            << ",\"grad_accum\":" << report.grad_accum
            << ",\"input_tokens\":" << report.input_tokens
            << ",\"loss_tokens\":" << report.loss_tokens
            << ",\"initial_loss\":" << report.initial_loss
            << ",\"loss\":" << report.loss
            << ",\"loss_finite\":true"
            << ",\"dense_cuda_path\":true"
            << ",\"weight_changed\":"
            << (report.weight_changed ? "true" : "false")
            << ",\"logits_checksum\":\""
            << lkjai::json_escape(report.logits_checksum) << "\""
            << ",\"config_path\":\""
            << lkjai::json_escape(report.config_path.string()) << "\""
            << ",\"packed_cache_path\":\""
            << lkjai::json_escape(report.packed_cache.string()) << "\""
            << ",\"checkpoint_path\":\""
            << lkjai::json_escape(report.checkpoint_dir.string()) << "\""
            << ",\"export_path\":\""
            << lkjai::json_escape(report.export_dir.string()) << "\""
            << ",\"timings\":{\"batch_load\":"
            << report.batch_load_seconds << ",\"forward\":"
            << report.forward_seconds << ",\"backward\":"
            << report.backward_seconds << ",\"optimizer\":"
            << report.optimizer_seconds << ",\"checkpoint\":"
            << report.checkpoint_seconds << ",\"export\":"
            << report.export_seconds << "}"
            << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
            << ",\"capability\":{" << lkjai::capability_json_fields(cuda)
            << "}"
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}
