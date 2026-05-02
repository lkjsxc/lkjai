#pragma once

#include <string>

#include "cuda_probe.hpp"

namespace lkjai {

std::string capability_json_fields(const CudaStatus& status);
std::string capability_json(const CudaStatus& status, bool ok);

}  // namespace lkjai
