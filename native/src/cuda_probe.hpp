#pragma once

#include <string>

namespace lkjai {

struct CudaStatus {
  bool available = false;
  std::string device;
  int compute_major = 0;
  int compute_minor = 0;
  bool bf16_supported = false;
  std::string warning;
};

CudaStatus cuda_status();

}  // namespace lkjai
