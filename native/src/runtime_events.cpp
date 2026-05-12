#include "runtime_events.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
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

struct RunSummary {
  std::string id;
  std::string created_at;
  std::string updated_at;
  int event_count = 0;
  std::string last_kind;
  std::string preview;
};

bool preview_kind(const std::string& kind) {
  return kind == "user" || kind == "assistant" || kind == "error";
}

RunSummary summarize_run(const std::filesystem::path& path) {
  RunSummary summary;
  summary.id = path.stem().string();
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    auto timestamp = json_first_string(line, "timestamp");
    auto kind = json_first_string(line, "kind");
    auto content = json_first_string(line, "content");
    if (summary.event_count == 0) summary.created_at = timestamp;
    summary.updated_at = timestamp;
    ++summary.event_count;
    summary.last_kind = kind;
    if (preview_kind(kind)) summary.preview = content;
  }
  return summary;
}

std::string run_summary_json(const RunSummary& run) {
  std::ostringstream out;
  out << "{\"run_id\":\"" << json_escape(run.id) << "\",\"created_at\":\""
      << json_escape(run.created_at) << "\",\"updated_at\":\""
      << json_escape(run.updated_at) << "\",\"event_count\":"
      << run.event_count << ",\"last_kind\":\"" << json_escape(run.last_kind)
      << "\",\"preview\":\"" << json_escape(run.preview) << "\"}";
  return out.str();
}

}  // namespace

std::string runtime_new_run_id() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  return "run-" + std::to_string(ms);
}

bool runtime_run_id_ok(const std::string& id) {
  if (id.empty() || id.size() > 96) return false;
  for (unsigned char ch : id) {
    if (!std::isalnum(ch) && ch != '_' && ch != '-') return false;
  }
  return true;
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

std::string runtime_chat_messages_json(const RuntimeConfig& cfg,
                                       const std::string& run_id,
                                       int limit) {
  std::ifstream file(runtime_run_path(cfg, run_id));
  std::vector<std::string> rows;
  std::string line;
  while (std::getline(file, line)) {
    auto kind = json_first_string(line, "kind");
    if (kind != "user" && kind != "assistant") continue;
    auto content = json_first_string(line, "content");
    rows.push_back("{\"role\":\"" + kind + "\",\"content\":\"" +
                   json_escape(content) + "\"}");
  }
  if (limit < 1) limit = 1;
  size_t start = rows.size() > static_cast<size_t>(limit)
                     ? rows.size() - static_cast<size_t>(limit)
                     : 0;
  std::ostringstream out;
  for (size_t i = start; i < rows.size(); ++i) {
    out << "," << rows[i];
  }
  return out.str();
}

std::string runtime_runs_json(const RuntimeConfig& cfg, int limit) {
  if (limit < 1) limit = 20;
  if (limit > 100) limit = 100;
  auto dir = std::filesystem::path(cfg.data_dir) / "agent" / "runs";
  std::vector<RunSummary> runs;
  if (std::filesystem::is_directory(dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (!entry.is_regular_file() || entry.path().extension() != ".jsonl") {
        continue;
      }
      runs.push_back(summarize_run(entry.path()));
    }
  }
  std::sort(runs.begin(), runs.end(),
            [](const RunSummary& a, const RunSummary& b) {
              return a.id > b.id;
            });
  std::ostringstream out;
  out << "{\"runs\":[";
  for (size_t i = 0; i < runs.size() && i < static_cast<size_t>(limit); ++i) {
    if (i > 0) out << ",";
    out << run_summary_json(runs[i]);
  }
  out << "]}";
  return out.str();
}

}  // namespace lkjai
