#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "cuda_probe.hpp"
#include "dense_model.hpp"
#include "env.hpp"
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

void export_smoke_artifacts(int steps) {
  auto data = std::filesystem::path(lkjai::env_string("DATA_DIR", "/tmp/lkjai-native-smoke"));
  auto model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
  if (!lkjai::write_dense_smoke_artifact(data / "exports" / model, steps, steps, true) ||
      !lkjai::write_dense_smoke_artifact(data.parent_path() / "models" / model,
                                         steps, steps, true)) {
    throw std::runtime_error("failed to write dense smoke artifact");
  }
}

}  // namespace

int main(int argc, char** argv) {
  int steps = int_arg(argc, argv, "--steps", 2);
  auto cuda = lkjai::cuda_status();
  if (has_flag(argc, argv, "--help")) {
    std::cout << "usage: lkjai-native-train --smoke [--steps N] | --train\n";
    return 0;
  }
  if (has_flag(argc, argv, "--train")) {
    return lkjai::run_corpus_training();
  }
  if (!has_flag(argc, argv, "--smoke")) {
    std::cerr << "native trainer requires --train or --smoke\n";
    return 2;
  }
  auto started = std::chrono::steady_clock::now();
  for (int step = 1; step <= steps; ++step) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::cerr << "{\"event\":\"native_train_smoke_step\",\"step\":" << step
              << "}\n";
  }
  export_smoke_artifacts(steps);
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "{\"status\":\"pass\",\"mode\":\"smoke\",\"steps\":" << steps
            << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}
