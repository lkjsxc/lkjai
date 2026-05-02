#pragma once

#include <string>

namespace lkjai {

struct DenseCudaCheck {
  bool ok = false;
  std::string device;
  std::string error;
};

DenseCudaCheck run_dense_cuda_check();

}  // namespace lkjai
