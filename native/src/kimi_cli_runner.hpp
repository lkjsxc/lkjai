#pragma once

#include <string>
#include <vector>

namespace lkjai::kimi_cli_runner {

struct RunnerConfig {
  std::string kimi_bin = "kimi";
  std::string agent_file;
  std::string base_url = "https://api.kimi.com/coding/v1";
  std::string model = "kimi-for-coding";
  std::string max_steps = "3";
  std::string max_tokens = "12000";
  std::string temperature = "0.2";
  int parallelism = 1;
  int max_retries = 0;
};

struct Job {
  std::string raw;
  std::string job_id;
  std::string input_jsonl;
  int ordinal = 0;
};

struct CallResult {
  std::string status = "fail";
  std::string text;
  std::string error;
  int attempts = 0;
  long long elapsed_ms = 0;
};

Job parse_job(const std::string& line);
CallResult run_job(const RunnerConfig& config, const Job& job);
std::string result_json(const Job& job, const CallResult& result);
void print_command_self_test(const RunnerConfig& config);

}  // namespace lkjai::kimi_cli_runner
