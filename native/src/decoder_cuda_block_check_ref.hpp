#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

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

std::vector<float> cpu_rope(const std::vector<float>& input, int batch,
                            int seq, int heads, int head_dim, float theta) {
  std::vector<float> out(input.size());
  for (int b = 0; b < batch; ++b)
    for (int s = 0; s < seq; ++s)
      for (int h = 0; h < heads; ++h)
        for (int p = 0; p < head_dim / 2; ++p) {
          size_t base = (((static_cast<size_t>(b) * seq + s) * heads + h) *
                         head_dim) +
                        p * 2;
          float x0 = f32(bf16(input[base]));
          float x1 = f32(bf16(input[base + 1]));
          float inv = std::pow(theta, -2.0f * static_cast<float>(p) /
                                          static_cast<float>(head_dim));
          float angle = static_cast<float>(s) * inv;
          out[base] = f32(bf16(x0 * std::cos(angle) - x1 * std::sin(angle)));
          out[base + 1] =
              f32(bf16(x0 * std::sin(angle) + x1 * std::cos(angle)));
        }
  return out;
}

bool close_enough(const std::vector<float>& got, const std::vector<float>& want,
                  double max_limit, double mean_limit,
                  const std::string& label) {
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

bool require_contains(const std::string& text, const std::string& needle) {
  if (text.find(needle) != std::string::npos) return true;
  std::cerr << "missing report field: " << needle << "\n";
  return false;
}

}  // namespace
