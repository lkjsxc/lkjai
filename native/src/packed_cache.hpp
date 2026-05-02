#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace lkjai {

struct PackedCacheStatus {
  bool ok = false;
  std::filesystem::path dir;
  uint64_t windows = 0;
  uint64_t tokens = 0;
  std::string error;
};

PackedCacheStatus inspect_packed_cache(const std::filesystem::path& dir);

}  // namespace lkjai
