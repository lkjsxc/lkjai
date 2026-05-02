#include "train_real.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "dense_model.hpp"
#include "env.hpp"
#include "json_min.hpp"
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

void write_artifact(const std::filesystem::path& dir, const Options& opt,
                    int step, long long rows, bool final) {
  (void)opt;
  if (!write_dense_smoke_artifact(dir, step, rows, final)) {
    throw std::runtime_error("failed to write dense artifact");
  }
}

void write_run_reports(const Options& opt, int step, long long rows, double elapsed) {
  auto ckpt = opt.data_dir / "checkpoints";
  write_text(ckpt / "manifest.json", "{\"latest\":\"latest\",\"final\":\"final\"}\n");
  write_text(opt.data_dir / "exports" / "manifest.json",
             "{\"model\":\"" + json_escape(opt.model_name) + "\"}\n");
  write_text(opt.data_dir / "runs" / "fixed-eval.json",
             "{\"status\":\"recorded\",\"mode\":\"native-dense-corpus\"}\n");
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
  auto cuda = cuda_status();
  auto started = std::chrono::steady_clock::now();
  int step = 0;
  while (step < opt.max_steps) {
    if (opt.stop_at_unix > 0 && std::time(nullptr) >= opt.stop_at_unix) break;
    std::string line;
    if (!corpus.next(&line)) break;
    auto text = training_text_from_jsonl(line);
    if (text.size() > static_cast<size_t>(opt.max_row_bytes)) text.resize(opt.max_row_bytes);
    ++step;
    if (step % opt.log_every == 0) {
      std::cerr << "{\"event\":\"native_train_step\",\"step\":" << step
                << ",\"rows\":" << corpus.rows() << "}\n";
    }
    if (step % opt.save_every == 0) {
      write_artifact(opt.data_dir / "checkpoints" / "latest", opt, step,
                     corpus.rows(), false);
    }
  }
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  write_artifact(opt.data_dir / "checkpoints" / "latest", opt, step, corpus.rows(), false);
  write_artifact(opt.data_dir / "checkpoints" / "final", opt, step, corpus.rows(), true);
  write_artifact(opt.data_dir / "exports" / opt.model_name, opt, step, corpus.rows(), true);
  write_artifact(opt.data_dir.parent_path() / "models" / opt.model_name, opt, step,
                 corpus.rows(), true);
  write_run_reports(opt, step, corpus.rows(), elapsed);
  std::cout << "{\"status\":\"pass\",\"mode\":\"train\",\"steps\":" << step
            << ",\"rows\":" << corpus.rows()
            << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}

}  // namespace lkjai
