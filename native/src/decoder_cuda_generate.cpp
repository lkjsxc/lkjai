#include "decoder_cuda_decode.hpp"

#include <algorithm>
#include <exception>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

int argmax(const std::vector<float>& logits) {
  return static_cast<int>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

int choose_next(const std::vector<float>& logits, const DecoderSampler& sampler,
                int step) {
  if (sampler.temperature <= 0.0f) return argmax(logits);
  return sample_next_token(logits, sampler.temperature, sampler.top_k,
                           sampler.top_p, sampler.seed, step);
}

}  // namespace

bool DecoderCudaInferenceSession::generate(
    const NativeTokenizer& tokenizer, const std::vector<uint16_t>& prompt_tokens,
    const DecoderSampler& sampler, DecoderKvCache* cache,
    DecoderCudaGenerateResult* result, std::string* error) {
  try {
    DecoderCudaGenerateResult local;
    local.sampler_backend = sampler.temperature <= 0.0f
                                ? "host_full_vocab"
                                : "cuda_topk_host_topp";
    int prefill = std::min<int>(prompt_tokens.size(), context_size());
    std::vector<uint16_t> window(prompt_tokens.end() - prefill,
                                 prompt_tokens.end());
    std::vector<float> current_logits;
    if (!logits(window, 0, false, cache, &current_logits, error)) return false;
    local.prefill_allocated_bytes = cache->allocated_bytes;
    local.cuda_kv_cache_used = cache->allocated_bytes > 0;
    int eos = tokenizer_id(tokenizer, "<eos>", tokenizer.eos_id);
    int end_action = tokenizer_id(tokenizer, "</action>", -1);
    for (int i = 0; i < sampler.max_tokens; ++i) {
      int next = choose_next(current_logits, sampler, i);
      local.generated.push_back(static_cast<uint16_t>(next));
      if (next == eos || next == end_action) {
        local.finish_reason = "stop";
        local.stop_reason = next == eos ? "eos" : "end_action";
        break;
      }
      if (i + 1 == sampler.max_tokens || cache->next_position[0] >= context_size()) {
        break;
      }
      DeviceAllocationStats before = device_allocation_stats();
      std::vector<uint16_t> one{static_cast<uint16_t>(next)};
      if (!logits(one, cache->next_position[0], true, cache, &current_logits,
                  error)) {
        return false;
      }
      DeviceAllocationStats after = device_allocation_stats();
      local.steady_state_token_allocations += static_cast<int>(
          device_allocation_count_delta(before, after));
      local.cuda_kv_cache_used = true;
    }
    local.workspace_bytes = workspace_bytes();
    *result = std::move(local);
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool decoder_cuda_generate(const TransformerState& state,
                           const NativeTokenizer& tokenizer,
                           const std::vector<uint16_t>& prompt_tokens,
                           const DecoderSampler& sampler,
                           DecoderKvCache* cache,
                           DecoderCudaGenerateResult* result,
                           std::string* error) {
  try {
    DecoderCudaInferenceSession session(state);
    return session.generate(tokenizer, prompt_tokens, sampler, cache, result,
                            error);
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace lkjai
