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

void cpu_rmsnorm_backward(const std::vector<float>& input,
                          const std::vector<float>& weight,
                          const std::vector<float>& dy, int rows, int hidden,
                          float eps, std::vector<float>* dx,
                          std::vector<float>* dw) {
  dx->assign(input.size(), 0.0f);
  dw->assign(weight.size(), 0.0f);
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    double dot = 0.0;
    for (int h = 0; h < hidden; ++h) {
      size_t i = static_cast<size_t>(r) * hidden + h;
      float x = f32(bf16(input[i]));
      float d = f32(bf16(dy[i]));
      ss += static_cast<double>(x) * x;
      dot += static_cast<double>(d) * weight[h] * x;
    }
    float inv = 1.0f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    float coeff = inv * inv * inv * static_cast<float>(dot) / hidden;
    for (int h = 0; h < hidden; ++h) {
      size_t i = static_cast<size_t>(r) * hidden + h;
      float x = f32(bf16(input[i]));
      float d = f32(bf16(dy[i]));
      (*dx)[i] = d * weight[h] * inv - x * coeff;
      (*dw)[h] += d * x * inv;
    }
  }
}

bool close(const std::vector<float>& got, const std::vector<float>& want,
           double max_limit, double mean_limit, const char* label) {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    double diff = std::abs(static_cast<double>(got[i]) - want[i]);
    max_abs = std::max(max_abs, diff);
    mean_abs += diff;
  }
  mean_abs /= static_cast<double>(got.size());
  if (max_abs <= max_limit && mean_abs <= mean_limit) return true;
  std::cerr << label << " parity failed max_abs=" << max_abs
            << " mean_abs=" << mean_abs << "\n";
  return false;
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
  std::vector<float> dy(input.size());
  for (size_t i = 0; i < dy.size(); ++i) {
    dy[i] = std::cos(static_cast<float>(i) * 0.13f) * 0.9f;
  }

  try {
    lkjai::CudaExecutionContext ctx;
    lkjai::DeviceTensor x({lkjai::DeviceDType::bf16, {rows, hidden}},
                          ctx.stream());
    lkjai::DeviceTensor w({lkjai::DeviceDType::f32, {hidden}},
                          ctx.stream());
    lkjai::DeviceTensor y({lkjai::DeviceDType::bf16, {rows, hidden}},
                          ctx.stream());
    lkjai::DeviceTensor d_y({lkjai::DeviceDType::bf16, {rows, hidden}},
                            ctx.stream());
    lkjai::DeviceTensor d_x({lkjai::DeviceDType::f32, {rows, hidden}},
                            ctx.stream());
    lkjai::DeviceTensor d_w({lkjai::DeviceDType::f32, {hidden}},
                            ctx.stream());
    x.copy_from_host_f32(input, ctx.stream());
    w.copy_from_host_f32(weight, ctx.stream());
    d_y.copy_from_host_f32(dy, ctx.stream());
    lkjai::decoder_launch_rmsnorm_bf16(x.data(),
                                       static_cast<const float*>(w.data()),
                                       y.data(), rows, hidden, eps,
                                       ctx.stream());
    auto got = y.copy_to_host_f32(ctx.stream());
    auto want = cpu_rmsnorm(input, weight, rows, hidden, eps);
    if (!close(got, want, 0.018, 0.004, "RMSNorm forward")) {
      return 1;
    }
    lkjai::decoder_launch_rmsnorm_backward_bf16(
        x.data(), static_cast<const float*>(w.data()), d_y.data(),
        static_cast<float*>(d_x.data()), static_cast<float*>(d_w.data()), rows,
        hidden, eps, 0.0f, ctx.stream());
    std::vector<float> want_dx;
    std::vector<float> want_dw;
    cpu_rmsnorm_backward(input, weight, dy, rows, hidden, eps, &want_dx,
                         &want_dw);
    if (!close(d_x.copy_to_host_f32(ctx.stream()), want_dx, 0.004, 0.001,
               "RMSNorm dX") ||
        !close(d_w.copy_to_host_f32(ctx.stream()), want_dw, 0.012, 0.004,
               "RMSNorm dW")) {
      return 1;
    }
    std::cout << "{\"status\":\"pass\",\"rows\":" << rows
              << ",\"hidden\":" << hidden
              << ",\"rmsnorm_backward\":true}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
