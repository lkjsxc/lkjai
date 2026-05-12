#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "decoder_cuda_block_check_ref.hpp"
#include "decoder_cuda_residual.hpp"
#include "runtime_device.hpp"

namespace {

std::vector<float> values(int n, float phase) {
  std::vector<float> out(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.19f + phase) * 0.8f;
  }
  return out;
}

std::vector<float> swiglu_grad_gate(const std::vector<float>& gate,
                                    const std::vector<float>& up,
                                    const std::vector<float>& dy) {
  std::vector<float> out(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    float g = f32(bf16(gate[i]));
    float u = f32(bf16(up[i]));
    float d = f32(bf16(dy[i]));
    float s = 1.0f / (1.0f + std::exp(-g));
    out[i] = f32(bf16(d * u * (s + g * s * (1.0f - s))));
  }
  return out;
}

std::vector<float> swiglu_grad_up(const std::vector<float>& gate,
                                  const std::vector<float>& dy) {
  std::vector<float> out(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    float g = f32(bf16(gate[i]));
    float d = f32(bf16(dy[i]));
    float s = 1.0f / (1.0f + std::exp(-g));
    out[i] = f32(bf16(d * g * s));
  }
  return out;
}

bool check_residual(lkjai::CudaExecutionContext* ctx) {
  constexpr int n = 113;
  auto dy = values(n, 0.3f);
  lkjai::DeviceTensor ddy({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  lkjai::DeviceTensor dl({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  lkjai::DeviceTensor dr({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  ddy.copy_from_host_f32(dy, ctx->stream());
  lkjai::decoder_launch_residual_add_backward_bf16(
      ddy.data(), dl.data(), dr.data(), n, ctx->stream());
  std::vector<float> want(dy.size());
  for (size_t i = 0; i < dy.size(); ++i) want[i] = f32(bf16(dy[i]));
  return close_enough(dl.copy_to_host_f32(ctx->stream()), want, 0.0, 0.0,
                      "residual d_lhs") &&
         close_enough(dr.copy_to_host_f32(ctx->stream()), want, 0.0, 0.0,
                      "residual d_rhs");
}

bool check_swiglu(lkjai::CudaExecutionContext* ctx) {
  constexpr int n = 129;
  auto gate = values(n, 0.1f);
  auto up = values(n, 0.7f);
  auto dy = values(n, 1.3f);
  lkjai::DeviceTensor dg({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  lkjai::DeviceTensor du({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  lkjai::DeviceTensor ddy({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  lkjai::DeviceTensor out_g({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  lkjai::DeviceTensor out_u({lkjai::DeviceDType::bf16, {n}}, ctx->stream());
  dg.copy_from_host_f32(gate, ctx->stream());
  du.copy_from_host_f32(up, ctx->stream());
  ddy.copy_from_host_f32(dy, ctx->stream());
  lkjai::decoder_launch_swiglu_backward_bf16(
      dg.data(), du.data(), ddy.data(), out_g.data(), out_u.data(), n,
      ctx->stream());
  return close_enough(out_g.copy_to_host_f32(ctx->stream()),
                      swiglu_grad_gate(gate, up, dy), 0.004, 0.001,
                      "SwiGLU d_gate") &&
         close_enough(out_u.copy_to_host_f32(ctx->stream()),
                      swiglu_grad_up(gate, dy), 0.004, 0.001,
                      "SwiGLU d_up");
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable: "
              << (cuda.error.empty() ? cuda.warning : cuda.error) << "\n";
    return 1;
  }
  try {
    lkjai::CudaExecutionContext ctx;
    if (!check_residual(&ctx) || !check_swiglu(&ctx)) return 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"decoder_backward_primitives\":true}\n";
  return 0;
}
