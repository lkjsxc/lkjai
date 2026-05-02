#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace lkjai {

bool write_dense_smoke_artifact(const std::filesystem::path& dir, int steps,
                                long long rows, bool final);
std::string dense_generate_action(const std::filesystem::path& model_dir,
                                  std::string_view prompt, int max_chars);

}  // namespace lkjai
