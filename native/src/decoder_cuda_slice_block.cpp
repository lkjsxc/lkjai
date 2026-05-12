#include "decoder_cuda_slice_internal.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#include "decoder_cuda_block.hpp"
#include "decoder_cuda_layer_forward.hpp"
#include "runtime_device.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

DeviceTensor bf16(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
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
    local.probe_batch = batch_size;
    local.probe_seq = seq;
    DeviceTensor x = bf16(ctx.stream(), rows, cfg.hidden_size);
    x.copy_from_host_f32(embedded_rows(state, batch), ctx.stream());
    DeviceWorkspace workspace(ctx.stream());
    DecoderCudaLayerForward forward(cfg, layer, &ctx, &workspace,
                                    kWorkspaceBytes);
    DeviceTensor out;
    forward.run(x, batch_size, seq, &out, &local);
    local.output_hidden = out.copy_to_host_f32(ctx.stream());
    local.output_rows = rows;
    local.output_hidden_size = cfg.hidden_size;
    local.projection_workspace_bytes = forward.workspace_high_water_bytes();
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
