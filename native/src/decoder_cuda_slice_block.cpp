#include "decoder_cuda_slice_internal.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#include "decoder_cuda_block.hpp"
#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_norm.hpp"
#include "decoder_cuda_residual.hpp"
#include "runtime_device.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

DeviceTensor bf16(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
}

bool finite(const DeviceTensor& tensor, cudaStream_t stream) {
  auto host = tensor.copy_to_host_f32(stream);
  return std::all_of(host.begin(), host.end(), [](float v) {
    return std::isfinite(v);
  });
}

std::vector<float> embedded_rows(const TransformerState& state,
                                 const PackedBatch& batch) {
  int hidden = state.cfg.hidden_size;
  std::vector<float> out(batch.tokens.size() * hidden);
  for (size_t row = 0; row < batch.tokens.size(); ++row) {
    int token = batch.tokens[row] % state.cfg.vocab_size;
    auto src = state.tok_embeddings.w.begin() + token * hidden;
    std::copy(src, src + hidden, out.begin() + row * hidden);
  }
  return out;
}

std::vector<float> projection_weight(const Parameter& p, int in, int out) {
  std::vector<float> t(static_cast<size_t>(in) * out);
  for (int i = 0; i < in; ++i) {
    for (int o = 0; o < out; ++o) {
      t[static_cast<size_t>(o) * in + i] = p.w[static_cast<size_t>(i) * out + o];
    }
  }
  return t;
}

}  // namespace

bool decoder_cuda_slice_run_block_forward(
    const TransformerState& state, const PackedBatch& batch,
    DecoderCudaForwardSubstrateReport* report, std::string* error) {
  DecoderCudaForwardSubstrateReport local;
  if (!decoder_cuda_block_shape(state.cfg, &local.shape, error)) return false;
  if (state.layers.empty()) {
    *error = "decoder block forward requires at least one layer";
    return false;
  }
  try {
    CudaExecutionContext ctx;
    const auto& cfg = state.cfg;
    const auto& layer = state.layers.front();
    int batch_size = batch.batch_size;
    int seq = batch.sequence_len;
    int rows = batch_size * seq;
    int kv_width = cfg.kv_heads * cfg.head_dim;
    local.probe_batch = batch_size;
    local.probe_seq = seq;
    DeviceTensor x = bf16(ctx.stream(), rows, cfg.hidden_size);
    x.copy_from_host_f32(embedded_rows(state, batch), ctx.stream());
    DeviceTensor norm = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor attn = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor q = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor k = bf16(ctx.stream(), rows, kv_width);
    DeviceTensor v = bf16(ctx.stream(), rows, kv_width);
    DeviceTensor o = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor attn_resid = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor mlp_norm = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor gate = bf16(ctx.stream(), rows, cfg.ffn_size);
    DeviceTensor up = bf16(ctx.stream(), rows, cfg.ffn_size);
    DeviceTensor swiglu = bf16(ctx.stream(), rows, cfg.ffn_size);
    DeviceTensor down = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor out = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor attn_w({DeviceDType::f32, {cfg.hidden_size}}, ctx.stream());
    DeviceTensor mlp_w({DeviceDType::f32, {cfg.hidden_size}}, ctx.stream());
    attn_w.copy_from_host_f32(layer.attn_norm.w, ctx.stream());
    mlp_w.copy_from_host_f32(layer.mlp_norm.w, ctx.stream());
    decoder_launch_rmsnorm_bf16(x.data(), static_cast<float*>(attn_w.data()),
                                norm.data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx.stream());
    local.rmsnorm_checked = finite(norm, ctx.stream());
    DeviceTensor wq = bf16(ctx.stream(), cfg.hidden_size, cfg.hidden_size);
    DeviceTensor wk = bf16(ctx.stream(), kv_width, cfg.hidden_size);
    DeviceTensor wv = bf16(ctx.stream(), kv_width, cfg.hidden_size);
    DeviceTensor wo = bf16(ctx.stream(), cfg.hidden_size, cfg.hidden_size);
    wq.copy_from_host_f32(projection_weight(layer.q_proj, cfg.hidden_size,
                                            cfg.hidden_size), ctx.stream());
    wk.copy_from_host_f32(projection_weight(layer.k_proj, cfg.hidden_size,
                                            kv_width), ctx.stream());
    wv.copy_from_host_f32(projection_weight(layer.v_proj, cfg.hidden_size,
                                            kv_width), ctx.stream());
    wo.copy_from_host_f32(projection_weight(layer.o_proj, cfg.hidden_size,
                                            cfg.hidden_size), ctx.stream());
    DeviceWorkspace workspace(ctx.stream());
    void* ws = workspace.allocate(kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), norm.data(),
                              wq.data(), q.data(), rows, cfg.hidden_size,
                              cfg.hidden_size, ws, kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), norm.data(),
                              wk.data(), k.data(), rows, cfg.hidden_size,
                              kv_width, ws, kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), norm.data(),
                              wv.data(), v.data(), rows, cfg.hidden_size,
                              kv_width, ws, kWorkspaceBytes);
    local.qkv_projection_checked =
        finite(q, ctx.stream()) && finite(k, ctx.stream()) &&
        finite(v, ctx.stream());
    decoder_launch_rope_bf16(q.data(), batch_size, seq, cfg.heads,
                             cfg.head_dim, cfg.rope_theta, ctx.stream());
    decoder_launch_rope_bf16(k.data(), batch_size, seq, cfg.kv_heads,
                             cfg.head_dim, cfg.rope_theta, ctx.stream());
    local.rope_checked = finite(q, ctx.stream()) && finite(k, ctx.stream());
    decoder_launch_causal_gqa_attention_bf16(
        q.data(), k.data(), v.data(), attn.data(), batch_size, seq, cfg.heads,
        cfg.kv_heads, cfg.head_dim, ctx.stream());
    local.attention_checked = finite(attn, ctx.stream());
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), attn.data(),
                              wo.data(), o.data(), rows, cfg.hidden_size,
                              cfg.hidden_size, ws, kWorkspaceBytes);
    local.o_projection_checked = finite(o, ctx.stream());
    decoder_launch_residual_add_bf16(x.data(), o.data(), attn_resid.data(),
                                     rows * cfg.hidden_size, ctx.stream());
    local.attention_residual_checked = finite(attn_resid, ctx.stream());
    decoder_launch_rmsnorm_bf16(attn_resid.data(),
                                static_cast<float*>(mlp_w.data()),
                                mlp_norm.data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx.stream());
    local.mlp_norm_checked = finite(mlp_norm, ctx.stream());
    DeviceTensor wg = bf16(ctx.stream(), cfg.ffn_size, cfg.hidden_size);
    DeviceTensor wu = bf16(ctx.stream(), cfg.ffn_size, cfg.hidden_size);
    DeviceTensor wd = bf16(ctx.stream(), cfg.hidden_size, cfg.ffn_size);
    wg.copy_from_host_f32(projection_weight(layer.gate_proj, cfg.hidden_size,
                                            cfg.ffn_size), ctx.stream());
    wu.copy_from_host_f32(projection_weight(layer.up_proj, cfg.hidden_size,
                                            cfg.ffn_size), ctx.stream());
    wd.copy_from_host_f32(projection_weight(layer.down_proj, cfg.ffn_size,
                                            cfg.hidden_size), ctx.stream());
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), mlp_norm.data(),
                              wg.data(), gate.data(), rows, cfg.hidden_size,
                              cfg.ffn_size, ws, kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), mlp_norm.data(),
                              wu.data(), up.data(), rows, cfg.hidden_size,
                              cfg.ffn_size, ws, kWorkspaceBytes);
    decoder_launch_swiglu_bf16(gate.data(), up.data(), swiglu.data(),
                               rows * cfg.ffn_size, ctx.stream());
    local.swiglu_checked = finite(swiglu, ctx.stream());
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), swiglu.data(),
                              wd.data(), down.data(), rows, cfg.ffn_size,
                              cfg.hidden_size, ws, kWorkspaceBytes);
    local.down_projection_checked = finite(down, ctx.stream());
    decoder_launch_residual_add_bf16(attn_resid.data(), down.data(),
                                     out.data(), rows * cfg.hidden_size,
                                     ctx.stream());
    local.block_residual_checked = finite(out, ctx.stream());
    local.outputs_finite = local.block_residual_checked;
    local.output_hidden = out.copy_to_host_f32(ctx.stream());
    local.output_rows = rows;
    local.output_hidden_size = cfg.hidden_size;
    local.projection_workspace_bytes = workspace.high_water_bytes();
    if (report) *report = local;
    if (local.outputs_finite) return true;
    *error = "decoder training block forward produced non-finite outputs";
    return false;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace lkjai
