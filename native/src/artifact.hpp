#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace lkjai {

struct ArtifactStatus {
  bool loaded = false;
  std::string model_name;
  std::filesystem::path model_dir;
  std::string error;
  std::vector<std::string> missing;
};

ArtifactStatus load_artifact(const std::filesystem::path& root,
                             const std::string& model_name);
bool inspect_artifact(const std::filesystem::path& model_dir,
                      std::string* error);

}  // namespace lkjai
