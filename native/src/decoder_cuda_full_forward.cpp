#include "decoder_cuda_block.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_layer_forward.hpp"
#include "decoder_cuda_norm.hpp"
#include "runtime_device.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

DeviceTensor bf16(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
}

bool finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float v) { return std::isfinite(v); });
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

void compare(const std::vector<float>& got, const std::vector<float>& want,
             double max_limit, double mean_limit, bool* close,
             double* max_abs, double* mean_abs) {
  *close = got.size() == want.size();
  *max_abs = 0.0;
  *mean_abs = 0.0;
  if (!*close || got.empty()) return;
  for (size_t i = 0; i < got.size(); ++i) {
    double diff = std::abs(static_cast<double>(got[i]) - want[i]);
    *max_abs = std::max(*max_abs, diff);
    *mean_abs += diff;
  }
  *mean_abs /= static_cast<double>(got.size());
  *close = *max_abs <= max_limit && *mean_abs <= mean_limit;
}

std::vector<float> last_row(const std::vector<float>& flat, int rows,
                            int width) {
  auto begin = flat.begin() + static_cast<size_t>(rows - 1) * width;
  return std::vector<float>(begin, begin + width);
}

}  // namespace

bool decoder_cuda_full_forward_probe(const TransformerState& state,
                                     const PackedBatch& batch,
                                     DecoderCudaFullForwardReport* report,
                                     std::string* error) {
  DecoderCudaFullForwardReport local;
  local.layers = state.cfg.layers;
  local.batch = batch.batch_size;
  local.sequence = batch.sequence_len;
  try {
    const auto& cfg = state.cfg;
    int rows = batch.batch_size * batch.sequence_len;
    CudaExecutionContext ctx;
    DeviceWorkspace workspace(ctx.stream());
    DeviceTensor hidden = bf16(ctx.stream(), rows, cfg.hidden_size);
    hidden.copy_from_host_f32(embedded_rows(state, batch), ctx.stream());
    local.layers_checked = true;
    for (const auto& layer : state.layers) {
      DecoderCudaForwardSubstrateReport layer_report;
      DecoderCudaLayerForward forward(cfg, layer, &ctx, &workspace,
                                      kWorkspaceBytes);
      DeviceTensor next;
      forward.run(hidden, batch.batch_size, batch.sequence_len, &next,
                  &layer_report);
      local.layers_checked = local.layers_checked && layer_report.outputs_finite;
      hidden = std::move(next);
    }
    DeviceTensor final = bf16(ctx.stream(), rows, cfg.hidden_size);
    DeviceTensor final_w({DeviceDType::f32, {cfg.hidden_size}}, ctx.stream());
    final_w.copy_from_host_f32(state.final_norm.w, ctx.stream());
    decoder_launch_rmsnorm_bf16(hidden.data(),
                                static_cast<float*>(final_w.data()),
                                final.data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx.stream());
    auto hidden_host = final.copy_to_host_f32(ctx.stream());
    local.final_norm_checked = finite(hidden_host);
    DeviceTensor lm_head = bf16(ctx.stream(), cfg.vocab_size, cfg.hidden_size);
    DeviceTensor logits = bf16(ctx.stream(), rows, cfg.vocab_size);
    lm_head.copy_from_host_f32(state.lm_head.w, ctx.stream());
    void* ws = workspace.allocate(kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), final.data(),
                              lm_head.data(), logits.data(), rows,
                              cfg.hidden_size, cfg.vocab_size, ws,
                              kWorkspaceBytes);
    auto logits_host = logits.copy_to_host_f32(ctx.stream());
    local.logits_checked = finite(logits_host);
    auto ref = transformer_forward(batch, state);
    compare(last_row(hidden_host, rows, cfg.hidden_size), ref.last_hidden,
            0.06, 0.015, &local.hidden_close, &local.hidden_max_abs,
            &local.hidden_mean_abs);
    compare(last_row(logits_host, rows, cfg.vocab_size), ref.next_logits,
            0.06, 0.015, &local.logits_close, &local.logits_max_abs,
            &local.logits_mean_abs);
    local.workspace_bytes = workspace.high_water_bytes();
    local.outputs_finite = local.layers_checked && local.final_norm_checked &&
                           local.logits_checked && local.hidden_close &&
                           local.logits_close;
    if (report) *report = local;
    if (local.outputs_finite) return true;
    *error = "decoder CUDA full forward parity probe failed";
    return false;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace lkjai
