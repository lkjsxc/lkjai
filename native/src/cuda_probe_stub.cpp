#include "cuda_probe.hpp"

namespace lkjai {

CudaStatus cuda_status() {
  return {false, "", "built without CUDA compiler"};
}

}  // namespace lkjai
