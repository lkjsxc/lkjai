#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_all(std::istream& in) {
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  return in ? read_all(in) : "";
}

std::string trim(std::string value) {
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
    value.pop_back();
  }
  while (!value.empty() && value.front() == ' ') value.erase(value.begin());
  return value;
}

std::string arg_value(int argc, char** argv, const std::string& key,
                      const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) return argv[i + 1];
  }
  return fallback;
}

std::string first_json_line(const std::string& text) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.front() == '{') return line;
  }
  return "";
}

std::string unescape_json_string(std::string_view value) {
  std::string out;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '\\' || i + 1 == value.size()) {
      out.push_back(value[i]);
      continue;
    }
    char next = value[++i];
    if (next == 'n') out.push_back('\n');
    else if (next == 't') out.push_back('\t');
    else if (next == 'r') out.push_back('\r');
    else out.push_back(next);
  }
  return out;
}

std::string json_content(const std::string& line) {
  const std::string key = "\"content\":\"";
  auto begin = line.find(key);
  if (begin == std::string::npos) return "";
  begin += key.size();
  bool escaped = false;
  for (size_t end = begin; end < line.size(); ++end) {
    char ch = line[end];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"') return unescape_json_string({line.data() + begin, end - begin});
  }
  return "";
}

int run_kimi(const std::string& kimi, const std::string& prompt,
             const std::string& max_steps, std::string* output) {
  int pipefd[2];
  if (pipe(pipefd) != 0) return 127;
  pid_t pid = fork();
  if (pid < 0) return 127;
  if (pid == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    std::vector<std::string> args = {
        kimi, "--print", "--output-format", "stream-json",
        "--final-message-only", "--max-steps-per-turn", max_steps,
        "--max-retries-per-step", "1", "--no-thinking", "-p", prompt};
    std::vector<char*> cargs;
    for (auto& arg : args) cargs.push_back(arg.data());
    cargs.push_back(nullptr);
    execvp(kimi.c_str(), cargs.data());
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
  return WIFEXITED(status) ? WEXITSTATUS(status) : 127;
}

}  // namespace

int main(int argc, char** argv) {
  auto key_file = arg_value(argc, argv, "--key-file");
  auto key = key_file.empty() ? std::string() : trim(read_file(key_file));
  if (!key.empty()) setenv("KIMI_API_KEY", key.c_str(), 1);
  setenv("KIMI_BASE_URL", arg_value(argc, argv, "--base-url",
                                    "https://api.kimi.com/coding/v1").c_str(), 1);
  setenv("KIMI_MODEL_NAME",
         arg_value(argc, argv, "--model", "kimi-for-coding").c_str(), 1);
  setenv("KIMI_CLI_NO_AUTO_UPDATE", "1", 1);
  auto path = std::string(std::getenv("PATH") ? std::getenv("PATH") : "");
  setenv("PATH", (std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                  "/.local/bin:" + path).c_str(), 1);
  auto prompt = read_all(std::cin);
  std::string output;
  int code = run_kimi(arg_value(argc, argv, "--kimi-bin", "kimi"), prompt,
                      arg_value(argc, argv, "--max-steps", "3"), &output);
  if (code != 0) {
    std::cerr << output;
    return code;
  }
  auto content = json_content(first_json_line(output));
  if (content.empty()) {
    std::cerr << output;
    return 2;
  }
  std::cout << content;
  return 0;
}
