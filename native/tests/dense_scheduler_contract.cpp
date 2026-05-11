#include <cmath>
#include <iostream>

#include "dense_cuda_internal.hpp"

namespace {

bool near(float actual, float expected, const char* label) {
  if (std::fabs(actual - expected) < 0.0001f) return true;
  std::cerr << label << " expected " << expected << " got " << actual << "\n";
  return false;
}

}  // namespace

int main() {
  lkjai::DenseTrainOptions opt;
  opt.lr = 1.0f;
  opt.warmup_steps = 2;
  opt.max_steps = 6;
  opt.lr_schedule = "warmup_cosine";
  opt.min_lr_fraction = 0.1f;
  bool ok = near(lkjai::dense_step_lr(opt, 1), 0.5f, "warmup") &&
            near(lkjai::dense_step_lr(opt, 2), 1.0f, "peak") &&
            near(lkjai::dense_step_lr(opt, 6), 0.1f, "floor");
  return ok ? 0 : 1;
}
