#pragma once

#include <string>

namespace lkjai {

struct CudaStatus {
  bool available = false;
  std::string device;
  std::string warning;
};

CudaStatus cuda_status();

}  // namespace lkjai
