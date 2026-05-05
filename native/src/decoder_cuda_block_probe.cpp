#include "decoder_cuda_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <vector>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_norm.hpp"
#include "decoder_cuda_residual.hpp"
#include "runtime_device.hpp"

namespace lkjai {
namespace {

constexpr size_t kProjectionWorkspaceBytes = 4 * 1024 * 1024;

bool finite_tensor(const DeviceTensor& tensor, cudaStream_t stream) {
  auto values = tensor.copy_to_host_f32(stream);
  return std::all_of(values.begin(), values.end(),
                     [](float v) { return std::isfinite(v); });
}

std::vector<float> deterministic_values(size_t count, float scale) {
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    float s = std::sin(static_cast<float>(i) * 0.071f);
    float c = std::cos(static_cast<float>(i) * 0.013f);
    out[i] = (s + 0.25f * c) * scale;
  }
  return out;
}

DeviceTensor bf16_tensor(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
}

void fill_tensor(DeviceTensor* tensor, size_t elements, float scale,
                 cudaStream_t stream) {
  tensor->copy_from_host_f32(deterministic_values(elements, scale), stream);
}

}  // namespace

bool decoder_cuda_forward_substrate_probe(
    const TransformerConfig& cfg, DecoderCudaForwardSubstrateReport* report,
    std::string* error) {
  DecoderCudaForwardSubstrateReport local;
  if (!decoder_cuda_block_shape(cfg, &local.shape, error)) return false;
  try {
    CudaExecutionContext ctx;
    int batch = 1;
    int seq = std::min(cfg.context, 2);
    int rows = batch * seq;
    local.probe_batch = batch;
    local.probe_seq = seq;
    DeviceTensor x = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor norm = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor rms_w({DeviceDType::f32, {cfg.hidden_size}}, ctx.stream());
    fill_tensor(&x, static_cast<size_t>(rows) * cfg.hidden_size, 0.125f,
                ctx.stream());
    std::vector<float> rms_weight(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i)
      rms_weight[i] = 0.8f + static_cast<float>(i % 29) * 0.01f;
    rms_w.copy_from_host_f32(rms_weight, ctx.stream());
    decoder_launch_rmsnorm_bf16(x.data(), static_cast<float*>(rms_w.data()),
                                norm.data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx.stream());
    local.rmsnorm_checked = finite_tensor(norm, ctx.stream());
    int kv_width = cfg.kv_heads * cfg.head_dim;
    DeviceTensor q = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor k = bf16_tensor(ctx.stream(), rows, kv_width);
    DeviceTensor v = bf16_tensor(ctx.stream(), rows, kv_width);
    DeviceTensor attn = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor o = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor attn_resid = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor wq = bf16_tensor(ctx.stream(), cfg.hidden_size, cfg.hidden_size);
    DeviceTensor wk = bf16_tensor(ctx.stream(), kv_width, cfg.hidden_size);
    DeviceTensor wv = bf16_tensor(ctx.stream(), kv_width, cfg.hidden_size);
    DeviceTensor wo = bf16_tensor(ctx.stream(), cfg.hidden_size, cfg.hidden_size);
    fill_tensor(&wq, static_cast<size_t>(cfg.hidden_size) * cfg.hidden_size,
                0.02f, ctx.stream());
    fill_tensor(&wk, static_cast<size_t>(kv_width) * cfg.hidden_size, 0.018f,
                ctx.stream());
    fill_tensor(&wv, static_cast<size_t>(kv_width) * cfg.hidden_size, 0.016f,
                ctx.stream());
    fill_tensor(&wo, static_cast<size_t>(cfg.hidden_size) * cfg.hidden_size,
                0.014f, ctx.stream());
    DeviceWorkspace workspace(ctx.stream());
    void* ws = workspace.allocate(kProjectionWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), norm.data(),
                              wq.data(), q.data(), rows, cfg.hidden_size,
                              cfg.hidden_size, ws, kProjectionWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), norm.data(),
                              wk.data(), k.data(), rows, cfg.hidden_size,
                              kv_width, ws, kProjectionWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), norm.data(),
                              wv.data(), v.data(), rows, cfg.hidden_size,
                              kv_width, ws, kProjectionWorkspaceBytes);
    local.qkv_projection_checked =
        finite_tensor(q, ctx.stream()) && finite_tensor(k, ctx.stream()) &&
        finite_tensor(v, ctx.stream());
    decoder_launch_rope_bf16(q.data(), batch, seq, cfg.heads, cfg.head_dim,
                             cfg.rope_theta, ctx.stream());
    decoder_launch_rope_bf16(k.data(), batch, seq, cfg.kv_heads, cfg.head_dim,
                             cfg.rope_theta, ctx.stream());
    local.rope_checked =
        finite_tensor(q, ctx.stream()) && finite_tensor(k, ctx.stream());
    decoder_launch_causal_gqa_attention_bf16(
        q.data(), k.data(), v.data(), attn.data(), batch, seq, cfg.heads,
        cfg.kv_heads, cfg.head_dim, ctx.stream());
    local.attention_checked = finite_tensor(attn, ctx.stream());
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), attn.data(),
                              wo.data(), o.data(), rows, cfg.hidden_size,
                              cfg.hidden_size, ws, kProjectionWorkspaceBytes);
    local.o_projection_checked = finite_tensor(o, ctx.stream());
    decoder_launch_residual_add_bf16(x.data(), o.data(), attn_resid.data(),
                                     rows * cfg.hidden_size, ctx.stream());
    local.attention_residual_checked = finite_tensor(attn_resid, ctx.stream());
    DeviceTensor mlp_norm = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor mlp_w({DeviceDType::f32, {cfg.hidden_size}}, ctx.stream());
    std::vector<float> mlp_weight(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i)
      mlp_weight[i] = 0.7f + static_cast<float>(i % 31) * 0.012f;
    mlp_w.copy_from_host_f32(mlp_weight, ctx.stream());
    decoder_launch_rmsnorm_bf16(attn_resid.data(),
                                static_cast<float*>(mlp_w.data()),
                                mlp_norm.data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx.stream());
    local.mlp_norm_checked = finite_tensor(mlp_norm, ctx.stream());
    DeviceTensor gate = bf16_tensor(ctx.stream(), rows, cfg.ffn_size);
    DeviceTensor up = bf16_tensor(ctx.stream(), rows, cfg.ffn_size);
    DeviceTensor swiglu = bf16_tensor(ctx.stream(), rows, cfg.ffn_size);
    DeviceTensor down = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor out = bf16_tensor(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor wg = bf16_tensor(ctx.stream(), cfg.ffn_size, cfg.hidden_size);
    DeviceTensor wu = bf16_tensor(ctx.stream(), cfg.ffn_size, cfg.hidden_size);
    DeviceTensor wd = bf16_tensor(ctx.stream(), cfg.hidden_size, cfg.ffn_size);
    fill_tensor(&wg, static_cast<size_t>(cfg.ffn_size) * cfg.hidden_size,
                0.015f, ctx.stream());
    fill_tensor(&wu, static_cast<size_t>(cfg.ffn_size) * cfg.hidden_size,
                0.017f, ctx.stream());
    fill_tensor(&wd, static_cast<size_t>(cfg.hidden_size) * cfg.ffn_size,
                0.013f, ctx.stream());
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), mlp_norm.data(),
                              wg.data(), gate.data(), rows, cfg.hidden_size,
                              cfg.ffn_size, ws, kProjectionWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), mlp_norm.data(),
                              wu.data(), up.data(), rows, cfg.hidden_size,
                              cfg.ffn_size, ws, kProjectionWorkspaceBytes);
    decoder_launch_swiglu_bf16(gate.data(), up.data(), swiglu.data(),
                               rows * cfg.ffn_size, ctx.stream());
    local.swiglu_checked = finite_tensor(swiglu, ctx.stream());
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), swiglu.data(),
                              wd.data(), down.data(), rows, cfg.ffn_size,
                              cfg.hidden_size, ws, kProjectionWorkspaceBytes);
    local.down_projection_checked = finite_tensor(down, ctx.stream());
    decoder_launch_residual_add_bf16(attn_resid.data(), down.data(),
                                     out.data(), rows * cfg.hidden_size,
                                     ctx.stream());
    local.block_residual_checked = finite_tensor(out, ctx.stream());
    local.outputs_finite = local.rmsnorm_checked && local.rope_checked &&
                           local.qkv_projection_checked &&
                           local.attention_checked &&
                           local.o_projection_checked &&
                           local.attention_residual_checked &&
                           local.mlp_norm_checked && local.swiglu_checked &&
                           local.down_projection_checked &&
                           local.block_residual_checked;
    local.projection_workspace_bytes = workspace.high_water_bytes();
    if (report) *report = local;
    if (local.outputs_finite) return true;
    *error = "decoder CUDA forward substrate produced non-finite outputs";
    return false;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace lkjai
