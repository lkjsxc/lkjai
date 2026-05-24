#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "kimi_cli_runner.hpp"

namespace {

std::string arg_value(int argc, char** argv, const std::string& key,
                      const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) return argv[i + 1];
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == key) return true;
  }
  return false;
}

int arg_int(int argc, char** argv, const std::string& key, int fallback) {
  auto value = arg_value(argc, argv, key);
  if (value.empty()) return fallback;
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

std::string read_all(std::istream& in) {
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace lkjai::kimi_cli_runner;
  RunnerConfig config;
  config.kimi_bin = arg_value(argc, argv, "--kimi-bin", config.kimi_bin);
  if (const char* kimi_bin = std::getenv("KIMI_CLI_BIN")) {
    if (*kimi_bin) config.kimi_bin = kimi_bin;
  }
  config.agent_file = arg_value(argc, argv, "--agent-file", "");
  if (const char* agent_file = std::getenv("KIMI_CLI_AGENT_FILE")) {
    if (*agent_file) config.agent_file = agent_file;
  }
  config.base_url = arg_value(argc, argv, "--base-url", config.base_url);
  config.model = arg_value(argc, argv, "--model", config.model);
  config.max_steps = arg_value(argc, argv, "--max-steps", config.max_steps);
  config.max_tokens = arg_value(argc, argv, "--max-tokens", config.max_tokens);
  config.temperature = arg_value(argc, argv, "--temperature", config.temperature);
  config.parallelism = std::max(1, arg_int(argc, argv, "--parallelism", 1));
  config.max_retries = std::max(0, arg_int(argc, argv, "--max-retries", 0));
  if (has_flag(argc, argv, "--self-test-command")) {
    print_command_self_test(config);
    return 0;
  }

  std::vector<Job> jobs;
  std::istringstream input(read_all(std::cin));
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) jobs.push_back(parse_job(line));
  }
  if (jobs.empty()) return 0;

  std::vector<std::string> outputs(jobs.size());
  std::atomic<size_t> next{0};
  const int workers = std::min<int>(config.parallelism, jobs.size());
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&]() {
      while (true) {
        const size_t index = next.fetch_add(1);
        if (index >= jobs.size()) break;
        outputs[index] = result_json(jobs[index], run_job(config, jobs[index]));
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (const auto& output : outputs) {
    std::cout << output << "\n";
  }
  return 0;
}
