#include "decoder_cuda_decode.hpp"

#include <memory>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_decode_kernels.hpp"
#include "decoder_cuda_layer_forward.hpp"
#include "decoder_cuda_norm.hpp"
#include "runtime_device.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

DeviceTensor bf16(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
}

void gather_embeddings(const DeviceTensor& table, const DeviceTensor& tokens,
                       DeviceTensor* out, int rows, int hidden, int vocab,
                       cudaStream_t stream) {
  decoder_cuda_gather_embeddings_bf16(table.data(), tokens.data(), out->data(),
                                      rows, hidden, vocab, stream);
}

void copy_bf16_to_host_f32(const DeviceTensor& src, DeviceTensor* temp,
                           std::vector<float>* out, cudaStream_t stream) {
  int n = static_cast<int>(src.spec().elements());
  decoder_cuda_bf16_to_f32(src.data(), temp->data(), n, stream);
  out->assign(static_cast<size_t>(n), 0.0f);
  require_cuda(cudaMemcpyAsync(out->data(), temp->data(),
                               static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "decoder logits D2H");
  require_cuda(cudaStreamSynchronize(stream), "decoder logits D2H sync");
}

}  // namespace

struct DecoderCudaInferenceSession::Impl {
  explicit Impl(const TransformerState& state)
      : state_(state),
        workspace_(ctx_.stream()),
        tok_embeddings_(bf16(ctx_.stream(), state.cfg.vocab_size,
                             state.cfg.hidden_size)),
        final_w_({DeviceDType::f32, {state.cfg.hidden_size}}, ctx_.stream()),
        lm_head_(bf16(ctx_.stream(), state.cfg.vocab_size,
                      state.cfg.hidden_size)),
        token_device_({DeviceDType::bf16, {state.cfg.context}}, ctx_.stream()),
        logits_f32_({DeviceDType::f32, {state.cfg.context, state.cfg.vocab_size}},
                    ctx_.stream()),
        one_hidden_a_(bf16(ctx_.stream(), 1, state.cfg.hidden_size)),
        one_hidden_b_(bf16(ctx_.stream(), 1, state.cfg.hidden_size)),
        one_final_(bf16(ctx_.stream(), 1, state.cfg.hidden_size)),
        one_logits_(bf16(ctx_.stream(), 1, state.cfg.vocab_size)) {
    tok_embeddings_.copy_from_host_f32(state.tok_embeddings.w, ctx_.stream());
    final_w_.copy_from_host_f32(state.final_norm.w, ctx_.stream());
    lm_head_.copy_from_host_f32(state.lm_head.w, ctx_.stream());
    for (const auto& layer : state.layers) {
      layers_.push_back(std::make_unique<DecoderCudaLayerForward>(
          state.cfg, layer, &ctx_, &workspace_, kWorkspaceBytes));
    }
  }

  bool logits(const std::vector<uint16_t>& tokens, int start_position,
              bool cached, DecoderKvCache* cache, std::vector<float>* out,
              std::string* error) {
    const auto& cfg = state_.cfg;
    int rows = static_cast<int>(tokens.size());
    if (rows <= 0 || rows > cfg.context) {
      *error = "decoder CUDA session token window out of bounds";
      return false;
    }
    DeviceAllocationStats before = device_allocation_stats();
    require_cuda(cudaMemcpyAsync(token_device_.data(), tokens.data(),
                                 tokens.size() * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, ctx_.stream()),
                 "decoder tokens H2D");
    DeviceTensor dynamic_hidden;
    DeviceTensor dynamic_hidden_b;
    DeviceTensor* hidden = nullptr;
    bool one_token = cached && rows == 1;
    if (one_token) {
      hidden = &one_hidden_a_;
    } else {
      dynamic_hidden = bf16(ctx_.stream(), rows, cfg.hidden_size);
      dynamic_hidden_b = bf16(ctx_.stream(), rows, cfg.hidden_size);
      hidden = &dynamic_hidden;
    }
    gather_embeddings(tok_embeddings_, token_device_, hidden, rows,
                      cfg.hidden_size, cfg.vocab_size, ctx_.stream());
    for (size_t i = 0; i < layers_.size(); ++i) {
      DecoderCudaLayerCacheView view{cache, static_cast<int>(i),
                                     start_position, cached};
      DeviceTensor* next = nullptr;
      if (one_token) {
        next = hidden == &one_hidden_a_ ? &one_hidden_b_ : &one_hidden_a_;
      } else {
        next = hidden == &dynamic_hidden ? &dynamic_hidden_b : &dynamic_hidden;
      }
      layers_[i]->run(*hidden, 1, rows, next, nullptr, &view);
      hidden = next;
    }
    DeviceTensor dynamic_final;
    DeviceTensor* final = &dynamic_final;
    if (one_token) {
      final = &one_final_;
    } else {
      dynamic_final = bf16(ctx_.stream(), rows, cfg.hidden_size);
    }
    decoder_launch_rmsnorm_bf16(hidden->data(),
                                static_cast<float*>(final_w_.data()),
                                final->data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx_.stream());
    DeviceTensor dynamic_logits;
    DeviceTensor* logits = &dynamic_logits;
    if (one_token) {
      logits = &one_logits_;
    } else {
      dynamic_logits = bf16(ctx_.stream(), rows, cfg.vocab_size);
    }
    void* ws = workspace_.allocate(kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx_.cublaslt(), ctx_.stream(), final->data(),
                              lm_head_.data(), logits->data(), rows,
                              cfg.hidden_size, cfg.vocab_size, ws,
                              kWorkspaceBytes);
    std::vector<float> all;
    copy_bf16_to_host_f32(*logits, &logits_f32_, &all, ctx_.stream());
    out->assign(all.end() - cfg.vocab_size, all.end());
    cache->next_position[0] = start_position + rows;
    DeviceAllocationStats after = device_allocation_stats();
    last_allocation_events_ =
        device_allocation_count_delta(before, after);
    return true;
  }

  uint64_t workspace_bytes() const { return workspace_.high_water_bytes(); }
  uint64_t last_allocation_events() const { return last_allocation_events_; }

  const TransformerState& state_;
  CudaExecutionContext ctx_;
  DeviceWorkspace workspace_;
  DeviceTensor tok_embeddings_;
  DeviceTensor final_w_;
  DeviceTensor lm_head_;
  DeviceTensor token_device_;
  DeviceTensor logits_f32_;
  DeviceTensor one_hidden_a_;
  DeviceTensor one_hidden_b_;
  DeviceTensor one_final_;
  DeviceTensor one_logits_;
  std::vector<std::unique_ptr<DecoderCudaLayerForward>> layers_;
  uint64_t last_allocation_events_ = 0;
};

DecoderCudaInferenceSession::DecoderCudaInferenceSession(
    const TransformerState& state)
    : impl_(std::make_unique<Impl>(state)) {}

DecoderCudaInferenceSession::DecoderCudaInferenceSession(
    DecoderCudaInferenceSession&&) noexcept = default;

DecoderCudaInferenceSession& DecoderCudaInferenceSession::operator=(
    DecoderCudaInferenceSession&&) noexcept = default;

DecoderCudaInferenceSession::~DecoderCudaInferenceSession() = default;

bool DecoderCudaInferenceSession::logits(
    const std::vector<uint16_t>& tokens, int start_position, bool cached,
    DecoderKvCache* cache, std::vector<float>* out, std::string* error) {
  return impl_->logits(tokens, start_position, cached, cache, out, error);
}

uint64_t DecoderCudaInferenceSession::workspace_bytes() const {
  return impl_->workspace_bytes();
}

int DecoderCudaInferenceSession::context_size() const {
  return impl_->state_.cfg.context;
}

}  // namespace lkjai
