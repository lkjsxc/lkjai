#include "decoder_cuda_slice_internal.hpp"

#include <vector>

#include "decoder_cuda_decode.hpp"
#include "decoder_decode.hpp"
#include "decoder_kv_cache.hpp"
#include "native_tokenizer.hpp"

namespace lkjai {

bool decoder_probe_generate_acceptance(const TransformerState& state,
                                       const NativeTokenizer& tokenizer,
                                       TransformerTrainReport* report,
                                       std::string* error) {
  DecoderCudaInferenceSession session(state);
  DecoderKvCache cache;
  DecoderKvCacheConfig cfg{state.cfg.layers, 1, state.cfg.kv_heads,
                           state.cfg.context, state.cfg.head_dim};
  if (!decoder_kv_cache_allocate(cfg, &cache, error)) return false;
  DecoderSampler sampler;
  sampler.max_tokens = 8;
  sampler.temperature = 0.0f;
  std::vector<uint16_t> prompt{1, 2, 3, 4};
  DecoderCudaGenerateResult result;
  if (!session.generate(tokenizer, prompt, sampler, &cache, &result, error)) {
    return false;
  }
  bool accepted_path =
      report->attention_backend == kDecoderAcceptedAttentionBackend;
  report->decode_supported = result.cuda_kv_cache_used;
  report->kv_cache_backend =
      accepted_path ? kDecoderAcceptedKvCacheBackend
                    : kDecoderRuntimePartialKvCacheBackend;
  report->decode_backend = accepted_path ? kDecoderAcceptedDecodeBackend
                                         : kDecoderRuntimePartialDecodeBackend;
  report->kv_cache_prefill_allocated_bytes = result.prefill_allocated_bytes;
  report->kv_cache_steady_state_token_allocations =
      result.steady_state_token_allocations;
  return true;
}

}  // namespace lkjai
