#include "runtime_events.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>

#include "json_min.hpp"

namespace lkjai {
namespace {

std::string timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

bool includes(const std::vector<std::string>& values, const std::string& value) {
  for (const auto& item : values) if (item == value) return true;
  return false;
}

}  // namespace

std::string runtime_new_run_id() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  return "run-" + std::to_string(ms);
}

std::filesystem::path runtime_run_path(const RuntimeConfig& cfg,
                                       const std::string& id) {
  return std::filesystem::path(cfg.data_dir) / "agent" / "runs" / (id + ".jsonl");
}

void runtime_append_event(const RuntimeConfig& cfg, const std::string& run_id,
                          const std::string& kind,
                          const std::string& content, int step,
                          const std::string& tool) {
  auto path = runtime_run_path(cfg, run_id);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::app);
  out << "{\"kind\":\"" << json_escape(kind) << "\",\"content\":\""
      << json_escape(content) << "\",\"timestamp\":\"" << timestamp() << "\"";
  if (step > 0) out << ",\"step\":" << step;
  if (!tool.empty()) out << ",\"tool\":\"" << json_escape(tool) << "\"";
  out << "}\n";
}

std::string runtime_events_json(const RuntimeConfig& cfg,
                                const std::string& run_id,
                                const std::vector<std::string>& visible) {
  std::ifstream file(runtime_run_path(cfg, run_id));
  std::ostringstream out;
  out << "[";
  std::string line;
  bool first = true;
  while (std::getline(file, line)) {
    auto kind = json_first_string(line, "kind");
    if (!visible.empty() && !includes(visible, kind)) continue;
    if (!first) out << ",";
    first = false;
    out << line;
  }
  out << "]";
  return out.str();
}

}  // namespace lkjai
