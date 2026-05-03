#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace lkjai {

struct DenseRuntimeTuning {
  std::string autotune_mode = "heuristic";
  std::string workspace_sweep = "4194304";
  std::string allocator_mode = "auto";
  std::string timing_mode = "deferred";
  std::vector<size_t> workspace_bytes = {4ull * 1024ull * 1024ull};

  bool autotune_enabled() const { return autotune_mode != "off"; }
  bool deferred_timing() const { return timing_mode == "deferred"; }
};

const DenseRuntimeTuning& dense_runtime_tuning();

}  // namespace lkjai
