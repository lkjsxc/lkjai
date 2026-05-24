#include "kimi_cli_runner.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "json_min.hpp"

namespace lkjai::kimi_cli_runner {
namespace {

std::string first_content_line(const std::string& text) {
  std::istringstream in(text);
  std::string line;
  std::string content;
  while (std::getline(in, line)) {
    if (!line.empty() && line.front() == '{') {
      auto next = json_first_string(line, "content");
      if (!next.empty()) content = next;
    }
  }
  return content;
}

std::filesystem::path write_temp_input(const std::string& job_id,
                                       const std::string& input) {
  auto safe = job_id.empty() ? std::string("job") : job_id;
  for (char& ch : safe) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' ||
          ch == '_')) {
      ch = '_';
    }
  }
  auto path = std::filesystem::temp_directory_path() /
              ("lkjai-kimi-runner-" + safe + "-" +
               std::to_string(::getpid()) + ".jsonl");
  std::ofstream out(path);
  out << input;
  if (!input.empty() && input.back() != '\n') out << '\n';
  return path;
}

std::vector<std::string> kimi_args(const RunnerConfig& config) {
  auto args = std::vector<std::string>{
      config.kimi_bin, "--print", "--input-format", "stream-json",
      "--output-format", "stream-json", "--final-message-only",
      "--max-ralph-iterations", "0", "--max-steps-per-turn",
      config.max_steps, "--no-thinking"};
  if (!config.agent_file.empty()) {
    args.push_back("--agent-file");
    args.push_back(config.agent_file);
  }
  return args;
}

int run_process(const RunnerConfig& config, const Job& job,
                std::string* output) {
  int pipefd[2];
  if (pipe(pipefd) != 0) return 127;
  auto input_path = write_temp_input(job.job_id, job.input_jsonl);
  pid_t pid = fork();
  if (pid < 0) {
    std::filesystem::remove(input_path);
    return 127;
  }
  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);
    std::freopen(input_path.string().c_str(), "r", stdin);
    setenv("KIMI_BASE_URL", config.base_url.c_str(), 1);
    setenv("KIMI_MODEL_NAME", config.model.c_str(), 1);
    setenv("KIMI_MODEL_MAX_TOKENS", config.max_tokens.c_str(), 1);
    setenv("KIMI_MODEL_TEMPERATURE", config.temperature.c_str(), 1);
    setenv("KIMI_CLI_NO_AUTO_UPDATE", "1", 1);
    auto path = std::string(std::getenv("PATH") ? std::getenv("PATH") : "");
    auto home = std::string(std::getenv("HOME") ? std::getenv("HOME") : "");
    setenv("PATH", (home + "/.local/bin:" + path).c_str(), 1);
    auto args = kimi_args(config);
    std::vector<char*> cargs;
    for (auto& arg : args) cargs.push_back(arg.data());
    cargs.push_back(nullptr);
    execvp(config.kimi_bin.c_str(), cargs.data());
    _exit(127);
  }
  close(pipefd[1]);
  char buffer[4096];
  ssize_t n = 0;
  while ((n = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
    output->append(buffer, static_cast<size_t>(n));
  }
  close(pipefd[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  std::filesystem::remove(input_path);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 127;
}

bool quota_like(const std::string& text) {
  return text.find("quota") != std::string::npos ||
         text.find("insufficient") != std::string::npos;
}

}  // namespace

CallResult run_job(const RunnerConfig& config, const Job& job) {
  CallResult result;
  auto start = std::chrono::steady_clock::now();
  const int max_attempts = std::max(1, config.max_retries + 1);
  for (int attempt = 1; attempt <= max_attempts; ++attempt) {
    result.attempts = attempt;
    std::string output;
    int code = run_process(config, job, &output);
    if (code == 0) {
      auto content = first_content_line(output);
      if (!content.empty()) {
        result.status = "pass";
        result.text = content;
        break;
      }
      result.error = output.empty() ? "kimi produced no stream-json content"
                                    : output.substr(0, 1000);
      break;
    }
    if (quota_like(output)) {
      result.status = "quota";
      result.error = output.substr(0, 1000);
      break;
    }
    result.status = (code == 75 && attempt < max_attempts) ? "retry" : "fail";
    result.error = output.empty() ? "kimi exited with code " + std::to_string(code)
                                  : output.substr(0, 1000);
    if (code != 75 || attempt >= max_attempts) break;
    std::this_thread::sleep_for(std::chrono::seconds(1 << (attempt - 1)));
  }
  auto end = std::chrono::steady_clock::now();
  result.elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return result;
}

std::string result_json(const Job& job, const CallResult& result) {
  std::ostringstream out;
  out << "{\"job_id\":\"" << json_escape(job.job_id)
      << "\",\"status\":\"" << json_escape(result.status)
      << "\",\"text\":\"" << json_escape(result.text)
      << "\",\"error\":\"" << json_escape(result.error)
      << "\",\"attempts\":" << result.attempts
      << ",\"elapsed_ms\":" << result.elapsed_ms
      << ",\"ordinal\":" << job.ordinal << "}";
  return out.str();
}

Job parse_job(const std::string& line) {
  Job job;
  job.raw = line;
  job.job_id = json_first_string(line, "job_id");
  job.input_jsonl = json_first_string(line, "input_jsonl");
  job.ordinal = json_int_value(line, "ordinal", 0);
  if (job.input_jsonl.empty()) job.input_jsonl = json_first_string(line, "prompt");
  if (job.job_id.empty()) job.job_id = "job-" + std::to_string(job.ordinal);
  return job;
}

void print_command_self_test(const RunnerConfig& config) {
  auto args = kimi_args(config);
  std::cout << "{\"argv\":[";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << "\"" << json_escape(args[i]) << "\"";
  }
  std::cout << "],\"uses_env_key\":true}\n";
}

}  // namespace lkjai::kimi_cli_runner
