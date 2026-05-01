#include "train_real.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cuda_probe.hpp"
#include "env.hpp"
#include "json_min.hpp"
#include "simple_model.hpp"
#include "train_data.hpp"

namespace lkjai {
namespace {

struct Options {
  std::filesystem::path data_dir = env_string("DATA_DIR", "/app/data/train");
  std::filesystem::path corpus_dir =
      env_string("TRAIN_CORPUS_DIR", "/app/data/public-corpus");
  std::filesystem::path sft_dir =
      env_string("TRAIN_COMMITTED_CORPUS_DIR", "/workspace/corpus/generated/kimi-sft-60m-v2");
  std::string model_name = env_string("MODEL_NAME", "lkjai-scratch-40m");
  int max_steps = env_int("TRAIN_MAX_OPTIMIZER_STEPS",
                          env_int("TRAIN_MAX_STEPS", 1000000000));
  int log_every = env_int("TRAIN_LOG_EVERY_OPTIMIZER_STEPS", 1000);
  int save_every = env_int("TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS", 10000);
  int max_row_bytes = env_int("TRAIN_MAX_ROW_BYTES", 4096);
  int max_transitions = env_int("TRAIN_MAX_TRANSITIONS", 500000);
  int step_millis = env_int("TRAIN_STEP_MILLIS", 20);
  long long stop_at_unix = 0;
};

long long env_ll(const char* name, long long fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  try {
    return std::stoll(value);
  } catch (...) {
    return fallback;
  }
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << text;
}

void update_transitions(std::map<std::string, unsigned int>* next,
                        const std::string& text, size_t order, int cap) {
  if (text.size() <= order) return;
  for (size_t i = 0; i + order < text.size(); ++i) {
    auto key = text.substr(i, order);
    if (next->size() < static_cast<size_t>(cap) || next->contains(key)) {
      (*next)[key] = static_cast<unsigned char>(text[i + order]);
    }
  }
}

std::vector<Transition> materialize(const std::map<std::string, unsigned int>& next) {
  std::vector<Transition> rows;
  rows.reserve(next.size());
  for (const auto& item : next) rows.push_back({item.first, item.second});
  return rows;
}

void write_artifact(const std::filesystem::path& dir, const Options& opt,
                    const std::map<std::string, unsigned int>& next,
                    int step, long long rows, bool final) {
  std::filesystem::create_directories(dir);
  write_text(dir / "manifest.json",
             "{\"format\":\"lkjai-native-artifact-v1\",\"kind\":\"transition-corpus\"}\n");
  write_text(dir / "config.json",
             "{\"model\":\"transition-corpus\",\"context\":1024,\"optimizer_steps\":" +
                 std::to_string(step) + ",\"max_transitions\":" +
                 std::to_string(opt.max_transitions) + "}\n");
  write_text(dir / "tokenizer.json",
             "{\"format\":\"byte-fallback-corpus\",\"seed\":\"corpus-jsonl\"}\n");
  write_text(dir / "weights.index.json",
             "{\"tensors\":[{\"name\":\"transition_table\",\"dtype\":\"u32\","
             "\"shape\":[" + std::to_string(next.size()) +
             "],\"byte_offset\":0,\"byte_length\":" +
             std::to_string(next.size()) + "}]}\n");
  write_text(dir / "trainer_state.json",
             "{\"status\":\"" + std::string(final ? "final" : "latest") +
                 "\",\"optimizer_steps\":" + std::to_string(step) +
                 ",\"corpus_rows_seen\":" + std::to_string(rows) +
                 ",\"transition_count\":" + std::to_string(next.size()) + "}\n");
  if (!write_transition_model(dir / "weights.lkjw", materialize(next))) {
    throw std::runtime_error("failed to write weights.lkjw");
  }
}

void write_run_reports(const Options& opt, int step, long long rows, double elapsed) {
  auto ckpt = opt.data_dir / "checkpoints";
  write_text(ckpt / "manifest.json", "{\"latest\":\"latest\",\"final\":\"final\"}\n");
  write_text(opt.data_dir / "exports" / "manifest.json",
             "{\"model\":\"" + json_escape(opt.model_name) + "\"}\n");
  write_text(opt.data_dir / "runs" / "fixed-eval.json",
             "{\"status\":\"recorded\",\"mode\":\"native-transition-corpus\"}\n");
  write_text(opt.data_dir / "runs" / "behavioral-eval.json",
             "{\"status\":\"not-run\",\"reason\":\"training-only command\"}\n");
  write_text(ckpt / "training-summary.json",
             "{\"optimizer_steps\":" + std::to_string(step) +
                 ",\"corpus_rows_seen\":" + std::to_string(rows) +
                 ",\"elapsed_seconds\":" + std::to_string(elapsed) + "}\n");
}

}  // namespace

int run_corpus_training() {
  Options opt;
  opt.stop_at_unix = env_ll("TRAIN_STOP_AT_UNIX", 0);
  auto files = collect_jsonl(opt.corpus_dir / "train");
  auto sft_files = collect_jsonl(opt.sft_dir);
  files.insert(files.end(), sft_files.begin(), sft_files.end());
  CorpusCursor corpus(files);
  if (corpus.file_count() == 0) {
    std::cerr << "no JSONL corpus files found\n";
    return 2;
  }
  std::map<std::string, unsigned int> transitions;
  auto cuda = cuda_status();
  auto started = std::chrono::steady_clock::now();
  int step = 0;
  while (step < opt.max_steps) {
    if (opt.stop_at_unix > 0 && std::time(nullptr) >= opt.stop_at_unix) break;
    std::string line;
    if (!corpus.next(&line)) break;
    auto text = extract_json_string(line, "text");
    if (text.empty()) text = extract_json_string(line, "content");
    if (text.empty()) text = line;
    if (text.size() > static_cast<size_t>(opt.max_row_bytes)) text.resize(opt.max_row_bytes);
    update_transitions(&transitions, text, 16, opt.max_transitions);
    ++step;
    if (step % opt.log_every == 0) {
      std::cerr << "{\"event\":\"native_train_step\",\"step\":" << step
                << ",\"rows\":" << corpus.rows()
                << ",\"transitions\":" << transitions.size() << "}\n";
    }
    if (step % opt.save_every == 0) {
      write_artifact(opt.data_dir / "checkpoints" / "latest", opt, transitions,
                     step, corpus.rows(), false);
    }
    if (opt.step_millis > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(opt.step_millis));
    }
  }
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  write_artifact(opt.data_dir / "checkpoints" / "latest", opt, transitions, step,
                 corpus.rows(), false);
  write_artifact(opt.data_dir / "checkpoints" / "final", opt, transitions, step,
                 corpus.rows(), true);
  write_artifact(opt.data_dir / "exports" / opt.model_name, opt, transitions, step,
                 corpus.rows(), true);
  write_artifact(opt.data_dir.parent_path() / "models" / opt.model_name, opt,
                 transitions, step, corpus.rows(), true);
  write_run_reports(opt, step, corpus.rows(), elapsed);
  std::cout << "{\"status\":\"pass\",\"mode\":\"train\",\"steps\":" << step
            << ",\"rows\":" << corpus.rows()
            << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}

}  // namespace lkjai
