#include "decoder_cuda_decode.hpp"

#include <algorithm>
#include <exception>
#include <memory>

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

int argmax(const std::vector<float>& logits) {
  return static_cast<int>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::vector<float> projection_weight(const Parameter& p, int in, int out) {
  std::vector<float> t(static_cast<size_t>(in) * out);
  for (int i = 0; i < in; ++i)
    for (int o = 0; o < out; ++o)
      t[static_cast<size_t>(o) * in + i] =
          p.w[static_cast<size_t>(i) * out + o];
  return t;
}

std::vector<float> embeddings(const TransformerState& state,
                              const std::vector<uint16_t>& tokens) {
  std::vector<float> out(tokens.size() * state.cfg.hidden_size);
  for (size_t row = 0; row < tokens.size(); ++row) {
    int token = tokens[row] % state.cfg.vocab_size;
    auto src = state.tok_embeddings.w.begin() + token * state.cfg.hidden_size;
    std::copy(src, src + state.cfg.hidden_size,
              out.begin() + row * state.cfg.hidden_size);
  }
  return out;
}

class DecodeRun {
 public:
  explicit DecodeRun(const TransformerState& state)
      : state_(state),
        workspace_(ctx_.stream()),
        final_w_({DeviceDType::f32, {state.cfg.hidden_size}}, ctx_.stream()),
        lm_head_(bf16(ctx_.stream(), state.cfg.vocab_size,
                      state.cfg.hidden_size)) {
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
    allocation_events_ += static_cast<uint64_t>(cfg.layers) + 4u;
    DeviceTensor hidden = bf16(ctx_.stream(), rows, cfg.hidden_size);
    hidden.copy_from_host_f32(embeddings(state_, tokens), ctx_.stream());
    for (size_t i = 0; i < layers_.size(); ++i) {
      DecoderCudaForwardSubstrateReport report;
      DecoderCudaLayerCacheView view{cache, static_cast<int>(i),
                                     start_position, cached};
      DeviceTensor next;
      layers_[i]->run(hidden, 1, rows, &next, &report, &view);
      hidden = std::move(next);
    }
    DeviceTensor final = bf16(ctx_.stream(), rows, cfg.hidden_size);
    decoder_launch_rmsnorm_bf16(hidden.data(),
                                static_cast<float*>(final_w_.data()),
                                final.data(), rows, cfg.hidden_size,
                                cfg.rms_norm_eps, ctx_.stream());
    DeviceTensor logits = bf16(ctx_.stream(), rows, cfg.vocab_size);
    void* ws = workspace_.allocate(kWorkspaceBytes);
    decoder_cuda_project_bf16(ctx_.cublaslt(), ctx_.stream(), final.data(),
                              lm_head_.data(), logits.data(), rows,
                              cfg.hidden_size, cfg.vocab_size, ws,
                              kWorkspaceBytes);
    auto all = logits.copy_to_host_f32(ctx_.stream());
    out->assign(all.end() - cfg.vocab_size, all.end());
    cache->next_position[0] = start_position + rows;
    return true;
  }

  uint64_t workspace_bytes() const { return workspace_.high_water_bytes(); }
  uint64_t allocation_events() const { return allocation_events_; }

 private:
  const TransformerState& state_;
  CudaExecutionContext ctx_;
  DeviceWorkspace workspace_;
  DeviceTensor final_w_;
  DeviceTensor lm_head_;
  std::vector<std::unique_ptr<DecoderCudaLayerForward>> layers_;
  uint64_t allocation_events_ = 0;
};

int choose_next(const std::vector<float>& logits, const DecoderSampler& sampler,
                int step) {
  if (sampler.temperature <= 0.0f) return argmax(logits);
  return sample_next_token(logits, sampler.temperature, sampler.top_k,
                           sampler.top_p, sampler.seed, step);
}

}  // namespace

bool decoder_cuda_generate(const TransformerState& state,
                           const NativeTokenizer& tokenizer,
                           const std::vector<uint16_t>& prompt_tokens,
                           const DecoderSampler& sampler,
                           DecoderKvCache* cache,
                           DecoderCudaGenerateResult* result,
                           std::string* error) {
  try {
    DecodeRun run(state);
    DecoderCudaGenerateResult local;
    int prefill = std::min<int>(prompt_tokens.size(), state.cfg.context);
    std::vector<uint16_t> window(prompt_tokens.end() - prefill,
                                 prompt_tokens.end());
    std::vector<float> logits;
    run.logits(window, 0, false, cache, &logits, error);
    local.prefill_allocated_bytes = cache->allocated_bytes;
    local.cuda_kv_cache_used = cache->allocated_bytes > 0;
    uint64_t prefill_events = run.allocation_events();
    int eos = tokenizer_id(tokenizer, "<eos>", tokenizer.eos_id);
    int end_action = tokenizer_id(tokenizer, "</action>", -1);
    for (int i = 0; i < sampler.max_tokens; ++i) {
      int next = choose_next(logits, sampler, i);
      local.generated.push_back(static_cast<uint16_t>(next));
      if (next == eos || next == end_action) {
        local.finish_reason = "stop";
        local.stop_reason = next == eos ? "eos" : "end_action";
        break;
      }
      if (i + 1 == sampler.max_tokens ||
          cache->next_position[0] >= state.cfg.context) {
        break;
      }
      std::vector<uint16_t> one{static_cast<uint16_t>(next)};
      run.logits(one, cache->next_position[0], true, cache, &logits, error);
      local.cuda_kv_cache_used = true;
    }
    local.steady_state_token_allocations =
        static_cast<int>(run.allocation_events() - prefill_events);
    local.workspace_bytes = run.workspace_bytes();
    *result = std::move(local);
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace lkjai
