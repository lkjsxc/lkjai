#pragma once

#include <string>

#include "dense_train.hpp"

namespace lkjai {

bool apply_training_config(const std::filesystem::path& path,
                           DenseTrainOptions* opt, std::string* error);

}  // namespace lkjai
