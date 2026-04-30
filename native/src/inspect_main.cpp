#include <filesystem>
#include <iostream>
#include <string>

#include "artifact.hpp"
#include "json_min.hpp"

namespace {

std::filesystem::path model_dir(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--model-dir") return argv[i + 1];
  }
  return "/models/lkjai-scratch-40m";
}

}  // namespace

int main(int argc, char** argv) {
  std::string error;
  auto dir = model_dir(argc, argv);
  if (!std::filesystem::is_directory(dir)) {
    std::cerr << "model dir not found: " << dir << "\n";
    return 1;
  }
  if (!lkjai::inspect_artifact(dir, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"model_dir\":\""
            << lkjai::json_escape(dir.string()) << "\"}\n";
  return 0;
}
