#include "dense_cuda_tuning.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace lkjai {
namespace {

std::string env_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value && *value ? value : fallback;
}

bool one_of(const std::string& value, std::initializer_list<const char*> items) {
  return std::any_of(items.begin(), items.end(), [&](const char* item) {
    return value == item;
  });
}

std::vector<size_t> parse_workspace(const std::string& text) {
  std::vector<size_t> values;
  std::stringstream in(text);
  std::string item;
  while (std::getline(in, item, ',')) {
    try {
      size_t value = static_cast<size_t>(std::stoull(item));
      values.push_back(value);
    } catch (...) {
    }
  }
  if (values.empty()) values.push_back(4ull * 1024ull * 1024ull);
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

DenseRuntimeTuning load_tuning() {
  DenseRuntimeTuning tuning;
  tuning.autotune_mode = env_or("LKJAI_DENSE_AUTOTUNE", "heuristic");
  if (!one_of(tuning.autotune_mode, {"heuristic", "benchmark", "off"})) {
    tuning.autotune_mode = "heuristic";
  }
  tuning.workspace_sweep = env_or("LKJAI_DENSE_WORKSPACE_SWEEP", "4194304");
  tuning.workspace_bytes = parse_workspace(tuning.workspace_sweep);
  tuning.allocator_mode = env_or("LKJAI_DENSE_ALLOCATOR", "auto");
  if (!one_of(tuning.allocator_mode, {"auto", "async", "legacy"})) {
    tuning.allocator_mode = "auto";
  }
  tuning.timing_mode = env_or("LKJAI_DENSE_TIMING", "deferred");
  if (!one_of(tuning.timing_mode, {"deferred", "legacy"})) {
    tuning.timing_mode = "deferred";
  }
  return tuning;
}

}  // namespace

const DenseRuntimeTuning& dense_runtime_tuning() {
  static DenseRuntimeTuning tuning = load_tuning();
  return tuning;
}

}  // namespace lkjai
