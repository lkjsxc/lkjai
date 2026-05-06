#include <iostream>
#include <string>

#include "json_min.hpp"
#include "packed_cache_build.hpp"
#include "packed_cache.hpp"

namespace {

std::string value(int argc, char** argv, const std::string& name) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return "";
}

bool flag(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::string command = argc > 1 ? argv[1] : "";
  if (command == "build") {
    lkjai::PackedCacheBuildOptions opt;
    opt.source = value(argc, argv, "--source");
    opt.tokenizer = value(argc, argv, "--tokenizer");
    opt.config = value(argc, argv, "--config");
    opt.out = value(argc, argv, "--out");
    opt.split = value(argc, argv, "--split");
    if (opt.split.empty()) opt.split = "train";
    opt.objective = value(argc, argv, "--objective");
    if (opt.objective.empty()) opt.objective = "causal_lm_full";
    auto seq = value(argc, argv, "--seq-len");
    if (!seq.empty()) opt.seq_len = std::stoi(seq);
    auto count = value(argc, argv, "--sequence-count");
    if (!count.empty()) opt.sequence_count = std::stoi(count);
    auto seed = value(argc, argv, "--seed");
    if (!seed.empty()) opt.seed = std::stoi(seed);
    opt.run_id = value(argc, argv, "--run-id");
    if (opt.run_id.empty()) opt.run_id = "native";
    std::string error;
    if (!lkjai::build_packed_cache(opt, &error)) {
      std::cerr << "packed cache build failed: " << error << "\n";
      return 2;
    }
    std::cout << "{\"status\":\"pass\",\"format\":\"lkjai-packed-cache\","
              << "\"out\":\"" << lkjai::json_escape(opt.out.string()) << "\"}\n";
    return 0;
  }
  if (command == "validate") {
    auto cache = value(argc, argv, "--cache");
    auto source = value(argc, argv, "--source");
    auto tokenizer = value(argc, argv, "--tokenizer");
    auto config = value(argc, argv, "--config");
    std::string error;
    if (cache.empty() || config.empty() ||
        !lkjai::validate_packed_cache_command(
            cache, source, tokenizer, config,
            flag(argc, argv, "--allow-smoke-fixture"), &error)) {
      std::cerr << "packed cache validation failed: " << error << "\n";
      return 2;
    }
    std::cout << "{\"status\":\"pass\",\"format\":\"lkjai-packed-cache\"}\n";
    return 0;
  }
  std::cerr << "usage: lkjai-native-packed-cache build|validate; "
               "validate accepts [--allow-smoke-fixture]\n";
  return 2;
}
