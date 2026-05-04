#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "decoder_cuda_norm.hpp"
#include "runtime_device.hpp"

namespace {

uint16_t bf16(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

float f32(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

std::vector<float> cpu_rmsnorm(const std::vector<float>& input,
                               const std::vector<float>& weight, int rows,
                               int hidden, float eps) {
  std::vector<float> out(input.size());
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int h = 0; h < hidden; ++h) {
      float v = f32(bf16(input[static_cast<size_t>(r) * hidden + h]));
      ss += static_cast<double>(v) * v;
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int h = 0; h < hidden; ++h) {
      size_t i = static_cast<size_t>(r) * hidden + h;
      out[i] = f32(bf16(f32(bf16(input[i])) * scale * weight[h]));
    }
  }
  return out;
}

}  // namespace

int main() {
  constexpr int rows = 5;
  constexpr int hidden = 96;
  constexpr float eps = 1.0e-5f;
  std::vector<float> input(static_cast<size_t>(rows) * hidden);
  std::vector<float> weight(hidden);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = std::sin(static_cast<float>(i) * 0.17f) * 1.7f +
               std::cos(static_cast<float>(i) * 0.031f) * 0.3f;
  }
  for (int h = 0; h < hidden; ++h) {
    weight[h] = 0.75f + static_cast<float>(h % 17) * 0.025f;
  }

  try {
    lkjai::CudaExecutionContext ctx;
    lkjai::DeviceTensor x({lkjai::DeviceDType::bf16, {rows, hidden}},
                          ctx.stream());
    lkjai::DeviceTensor w({lkjai::DeviceDType::f32, {hidden}},
                          ctx.stream());
    lkjai::DeviceTensor y({lkjai::DeviceDType::bf16, {rows, hidden}},
                          ctx.stream());
    x.copy_from_host_f32(input, ctx.stream());
    w.copy_from_host_f32(weight, ctx.stream());
    lkjai::decoder_launch_rmsnorm_bf16(x.data(),
                                       static_cast<const float*>(w.data()),
                                       y.data(), rows, hidden, eps,
                                       ctx.stream());
    auto got = y.copy_to_host_f32(ctx.stream());
    auto want = cpu_rmsnorm(input, weight, rows, hidden, eps);
    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      double diff = std::abs(static_cast<double>(got[i]) - want[i]);
      max_abs = std::max(max_abs, diff);
      mean_abs += diff;
    }
    mean_abs /= static_cast<double>(got.size());
    if (max_abs > 0.018 || mean_abs > 0.004) {
      std::cerr << "RMSNorm parity failed max_abs=" << max_abs
                << " mean_abs=" << mean_abs << "\n";
      return 1;
    }
    std::cout << "{\"status\":\"pass\",\"rows\":" << rows
              << ",\"hidden\":" << hidden << ",\"max_abs_diff\":" << max_abs
              << ",\"mean_abs_diff\":" << mean_abs << "}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
