#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "decoder_cuda_block.hpp"
#include "runtime_device.hpp"

namespace {

uint16_t bf16(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

float f32(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

std::vector<float> cpu_rope_backward(const std::vector<float>& dy, int batch,
                                     int seq, int heads, int head_dim,
                                     int offset, float theta) {
  std::vector<float> out(dy.size());
  for (int b = 0; b < batch; ++b)
    for (int s_pos = 0; s_pos < seq; ++s_pos)
      for (int h = 0; h < heads; ++h)
        for (int p = 0; p < head_dim / 2; ++p) {
          size_t base = (((static_cast<size_t>(b) * seq + s_pos) * heads + h) *
                         head_dim) +
                        p * 2;
          float g0 = f32(bf16(dy[base]));
          float g1 = f32(bf16(dy[base + 1]));
          float inv = std::pow(theta, -2.0f * static_cast<float>(p) /
                                          static_cast<float>(head_dim));
          float angle = static_cast<float>(offset + s_pos) * inv;
          out[base] = f32(bf16(g0 * std::cos(angle) + g1 * std::sin(angle)));
          out[base + 1] =
              f32(bf16(-g0 * std::sin(angle) + g1 * std::cos(angle)));
        }
  return out;
}

bool close(const std::vector<float>& got, const std::vector<float>& want) {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    double diff = std::abs(static_cast<double>(got[i]) - want[i]);
    max_abs = std::max(max_abs, diff);
    mean_abs += diff;
  }
  mean_abs /= static_cast<double>(got.size());
  if (max_abs <= 0.004 && mean_abs <= 0.001) return true;
  std::cerr << "RoPE backward parity failed max_abs=" << max_abs
            << " mean_abs=" << mean_abs << "\n";
  return false;
}

}  // namespace

int main() {
  constexpr int batch = 2;
  constexpr int seq = 5;
  constexpr int heads = 3;
  constexpr int head_dim = 8;
  constexpr int offset = 7;
  constexpr float theta = 10000.0f;
  std::vector<float> dy(batch * seq * heads * head_dim);
  for (size_t i = 0; i < dy.size(); ++i) {
    dy[i] = std::sin(static_cast<float>(i) * 0.19f) * 0.8f;
  }
  try {
    lkjai::CudaExecutionContext ctx;
    lkjai::DeviceTensor d_y({lkjai::DeviceDType::bf16,
                             {batch, seq, heads, head_dim}},
                            ctx.stream());
    lkjai::DeviceTensor d_x({lkjai::DeviceDType::bf16,
                             {batch, seq, heads, head_dim}},
                            ctx.stream());
    d_y.copy_from_host_f32(dy, ctx.stream());
    lkjai::decoder_launch_rope_backward_bf16_at(
        d_y.data(), d_x.data(), batch, seq, heads, head_dim, offset, theta,
        ctx.stream());
    if (!close(d_x.copy_to_host_f32(ctx.stream()),
               cpu_rope_backward(dy, batch, seq, heads, head_dim, offset,
                                 theta))) {
      return 1;
    }
    std::cout << "{\"status\":\"pass\",\"rope_backward\":true}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
