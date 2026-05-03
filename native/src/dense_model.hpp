#pragma once

#include <filesystem>

namespace lkjai {

bool write_dense_smoke_artifact(const std::filesystem::path& dir, int steps,
                                long long rows, bool final);

}  // namespace lkjai
