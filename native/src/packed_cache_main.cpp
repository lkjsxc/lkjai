#include <iostream>
#include <string>

#include "json_min.hpp"
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
  if (!flag(argc, argv, "--migrate-v1-to-v2")) {
    std::cerr << "usage: lkjai-native-packed-cache --migrate-v1-to-v2 "
                 "--in DIR --out DIR --config FILE [--link-mode hardlink]\n";
    return 2;
  }
  auto in = value(argc, argv, "--in");
  auto out = value(argc, argv, "--out");
  auto config = value(argc, argv, "--config");
  auto link_mode = value(argc, argv, "--link-mode");
  if (link_mode.empty()) link_mode = "hardlink";
  std::string error;
  if (in.empty() || out.empty() || config.empty() ||
      !lkjai::migrate_packed_cache_v1_to_v2(in, out, config, link_mode, &error)) {
    std::cerr << "packed cache migration failed: " << error << "\n";
    return 2;
  }
  std::cout << "{\"status\":\"pass\",\"format\":\"lkjai-packed-cache-v2\","
            << "\"out\":\"" << lkjai::json_escape(out) << "\"}\n";
  return 0;
}
