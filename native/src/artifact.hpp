#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace lkjai {

struct ArtifactStatus {
  bool loaded = false;
  std::string model_name;
  std::filesystem::path model_dir;
  std::string kind;
  std::string storage_kind;
  std::string error;
  std::vector<std::string> missing;
};

ArtifactStatus load_artifact(const std::filesystem::path& root,
                             const std::string& model_name);
bool inspect_artifact(const std::filesystem::path& model_dir,
                      std::string* error);
std::string artifact_logits_checksum(const std::filesystem::path& model_dir);
std::string artifact_text_checksum(std::string_view text);

}  // namespace lkjai
