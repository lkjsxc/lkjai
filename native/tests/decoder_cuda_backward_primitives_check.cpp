#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "decoder_cuda_block_check_ref.hpp"
#include "decoder_cuda_block_internal.hpp"
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

std::vector<float> projection_dx(const std::vector<float>& dy,
                                 const std::vector<float>& w, int rows,
                                 int in, int out) {
  std::vector<float> dx(static_cast<size_t>(rows) * in);
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < in; ++i) {
      float sum = 0.0f;
      for (int o = 0; o < out; ++o) {
        sum += f32(bf16(dy[static_cast<size_t>(r) * out + o])) *
               f32(bf16(w[static_cast<size_t>(o) * in + i]));
      }
      dx[static_cast<size_t>(r) * in + i] = sum;
    }
  }
  return dx;
}

std::vector<float> projection_dw(const std::vector<float>& x,
                                 const std::vector<float>& dy, int rows,
                                 int in, int out) {
  std::vector<float> dw(static_cast<size_t>(out) * in);
  for (int o = 0; o < out; ++o) {
    for (int i = 0; i < in; ++i) {
      float sum = 0.0f;
      for (int r = 0; r < rows; ++r) {
        sum += f32(bf16(dy[static_cast<size_t>(r) * out + o])) *
               f32(bf16(x[static_cast<size_t>(r) * in + i]));
      }
      dw[static_cast<size_t>(o) * in + i] = sum;
    }
  }
  return dw;
}

bool check_projection_backward(lkjai::CudaExecutionContext* ctx) {
  constexpr int rows = 5;
  constexpr int in = 7;
  constexpr int out = 6;
  auto x = values(rows * in, 0.2f);
  auto w = values(out * in, 0.6f);
  auto dy = values(rows * out, 1.1f);
  lkjai::DeviceTensor dx({lkjai::DeviceDType::f32, {rows, in}},
                         ctx->stream());
  lkjai::DeviceTensor dw({lkjai::DeviceDType::f32, {out, in}},
                         ctx->stream());
  lkjai::DeviceTensor x_dev({lkjai::DeviceDType::bf16, {rows, in}},
                            ctx->stream());
  lkjai::DeviceTensor w_dev({lkjai::DeviceDType::bf16, {out, in}},
                            ctx->stream());
  lkjai::DeviceTensor dy_dev({lkjai::DeviceDType::bf16, {rows, out}},
                             ctx->stream());
  x_dev.copy_from_host_f32(x, ctx->stream());
  w_dev.copy_from_host_f32(w, ctx->stream());
  dy_dev.copy_from_host_f32(dy, ctx->stream());
  lkjai::decoder_cuda_project_backward_bf16(
      ctx->cublaslt(), ctx->stream(), x_dev.data(), w_dev.data(),
      dy_dev.data(), dx.data(), dw.data(), rows, in, out, nullptr, 0, 0.0f);
  return close_enough(dx.copy_to_host_f32(ctx->stream()),
                      projection_dx(dy, w, rows, in, out), 0.006, 0.002,
                      "projection dX") &&
         close_enough(dw.copy_to_host_f32(ctx->stream()),
                      projection_dw(x, dy, rows, in, out), 0.006, 0.002,
                      "projection dW");
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
    if (!check_residual(&ctx) || !check_swiglu(&ctx) ||
        !check_projection_backward(&ctx)) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"decoder_backward_primitives\":true}\n";
  return 0;
}
