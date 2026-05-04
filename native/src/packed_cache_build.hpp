#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

struct PackedCacheBuildOptions {
  std::filesystem::path source;
  std::filesystem::path tokenizer;
  std::filesystem::path config;
  std::filesystem::path out;
  std::string split = "train";
  std::string objective = "causal_lm_full";
  int seq_len = 1024;
  int sequence_count = 0;
  int seed = 0;
  std::string run_id = "native";
};

bool build_packed_cache(const PackedCacheBuildOptions& opt, std::string* error);
bool validate_packed_cache_command(const std::filesystem::path& cache,
                                   const std::filesystem::path& config,
                                   std::string* error);

}  // namespace lkjai
