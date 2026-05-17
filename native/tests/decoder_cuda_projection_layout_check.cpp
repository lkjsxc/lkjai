#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_block_check_ref.hpp"
#include "decoder_cuda_block_internal.hpp"
#include "runtime_device.hpp"

namespace {

std::vector<float> values(int n, float phase) {
  std::vector<float> out(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = std::sin(i * 0.17f + phase);
  return out;
}

std::vector<float> forward_weight(const std::vector<float>& param, int in, int out) {
  std::vector<float> w(static_cast<size_t>(out) * in);
  for (int i = 0; i < in; ++i) {
    for (int o = 0; o < out; ++o) w[static_cast<size_t>(o) * in + i] = param[static_cast<size_t>(i) * out + o];
  }
  return w;
}

std::vector<float> ref_dw_param(const std::vector<float>& x,
                                const std::vector<float>& dy, int rows,
                                int in, int out) {
  std::vector<float> dw(static_cast<size_t>(in) * out);
  for (int i = 0; i < in; ++i) {
    for (int o = 0; o < out; ++o) {
      float sum = 0.0f;
      for (int r = 0; r < rows; ++r) {
        sum += f32(bf16(x[static_cast<size_t>(r) * in + i])) *
               f32(bf16(dy[static_cast<size_t>(r) * out + o]));
      }
      dw[static_cast<size_t>(i) * out + o] = sum;
    }
  }
  return dw;
}

bool check_shape(lkjai::CudaExecutionContext* ctx, int rows, int in, int out,
                 const char* label) {
  auto x = values(rows * in, 0.2f);
  auto param = values(in * out, 0.6f);
  auto dy = values(rows * out, 1.1f);
  auto w = forward_weight(param, in, out);
  lkjai::DeviceTensor x_dev({lkjai::DeviceDType::bf16, {rows, in}}, ctx->stream());
  lkjai::DeviceTensor w_dev({lkjai::DeviceDType::bf16, {out, in}}, ctx->stream());
  lkjai::DeviceTensor dy_dev({lkjai::DeviceDType::bf16, {rows, out}}, ctx->stream());
  lkjai::DeviceTensor dx({lkjai::DeviceDType::f32, {rows, in}}, ctx->stream());
  lkjai::DeviceTensor dw({lkjai::DeviceDType::f32, {in, out}}, ctx->stream());
  x_dev.copy_from_host_f32(x, ctx->stream());
  w_dev.copy_from_host_f32(w, ctx->stream());
  dy_dev.copy_from_host_f32(dy, ctx->stream());
  lkjai::decoder_cuda_project_backward_param_layout_bf16(
      ctx->cublaslt(), ctx->stream(), x_dev.data(), w_dev.data(),
      dy_dev.data(), dx.data(), dw.data(), rows, in, out, nullptr, 0, 0.0f);
  return close_enough(dw.copy_to_host_f32(ctx->stream()),
                      ref_dw_param(x, dy, rows, in, out), 0.006, 0.002, label);
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable\n";
    return 1;
  }
  try {
    lkjai::CudaExecutionContext ctx;
    if (!check_shape(&ctx, 5, 7, 6, "square-ish") ||
        !check_shape(&ctx, 4, 32, 16, "hidden-to-kv") ||
        !check_shape(&ctx, 4, 32, 64, "hidden-to-ffn") ||
        !check_shape(&ctx, 4, 64, 32, "ffn-to-hidden") ||
        !check_shape(&ctx, 4, 32, 32, "square")) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"projection_param_layout\":true}\n";
  return 0;
}
