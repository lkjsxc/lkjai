#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "cuda_probe.hpp"
#include "env.hpp"
#include "simple_model.hpp"
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

const char* kSmokeAction =
    "<action>\n"
    "<reasoning>The native smoke model completed a trained transition decode.</reasoning>\n"
    "<tool>agent.finish</tool>\n"
    "<content>native smoke complete</content>\n"
    "</action>";

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << text;
}

void write_artifact(const std::filesystem::path& dir, int steps) {
  std::filesystem::create_directories(dir);
  write_text(dir / "manifest.json",
             "{\"format\":\"lkjai-native-artifact-v1\",\"kind\":\"transition-smoke\"}\n");
  write_text(dir / "config.json",
             "{\"model\":\"transition-smoke\",\"context\":1024,\"steps\":" +
                 std::to_string(steps) + "}\n");
  write_text(dir / "tokenizer.json",
             "{\"format\":\"byte-fallback-smoke\",\"seed\":\"<action>\"}\n");
  auto transitions = lkjai::train_transitions(kSmokeAction);
  if (!lkjai::write_transition_model(dir / "weights.lkjw", transitions)) {
    throw std::runtime_error("failed to write weights.lkjw");
  }
  auto weight_bytes = std::filesystem::file_size(dir / "weights.lkjw");
  write_text(dir / "weights.index.json",
             "{\"tensors\":[{\"name\":\"transition_table\",\"dtype\":\"u32\","
             "\"shape\":[1],\"byte_offset\":0,\"byte_length\":" +
                 std::to_string(weight_bytes) + "}]}\n");
  write_text(dir / "trainer_state.json",
             "{\"status\":\"smoke-trained\",\"optimizer_steps\":" +
                 std::to_string(steps) + "}\n");
}

void export_smoke_artifacts(int steps) {
  auto data = std::filesystem::path(lkjai::env_string("DATA_DIR", "/tmp/lkjai-native-smoke"));
  auto model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
  write_artifact(data / "exports" / model, steps);
  write_artifact(data.parent_path() / "models" / model, steps);
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
