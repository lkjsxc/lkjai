#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace lkjai {

bool validate_dense_weight_index(std::string_view text, uint64_t weight_bytes,
                                 std::string* error);
bool validate_transformer_weight_index(std::string_view text,
                                       std::string_view config,
                                       uint64_t weight_bytes,
                                       std::string* error);
bool validate_dense_optimizer(const std::filesystem::path& model_dir,
                              std::string* error);

}  // namespace lkjai
