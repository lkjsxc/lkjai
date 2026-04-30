#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "cuda_probe.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  int steps = int_arg(argc, argv, "--steps", 2);
  auto cuda = lkjai::cuda_status();
  if (!has_flag(argc, argv, "--smoke")) {
    std::cerr << "native trainer currently requires --smoke\n";
    return 2;
  }
  auto started = std::chrono::steady_clock::now();
  for (int step = 1; step <= steps; ++step) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::cerr << "{\"event\":\"native_train_smoke_step\",\"step\":" << step
              << "}\n";
  }
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "{\"status\":\"pass\",\"mode\":\"smoke\",\"steps\":" << steps
            << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}
