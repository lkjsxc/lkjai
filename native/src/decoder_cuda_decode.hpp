#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "decoder_chat_request.hpp"
#include "decoder_kv_cache.hpp"
#include "native_tokenizer.hpp"
#include "transformer_state.hpp"

namespace lkjai {

struct DecoderCudaGenerateResult {
  std::vector<uint16_t> generated;
  std::string finish_reason = "length";
  std::string stop_reason = "max_tokens";
  uint64_t prefill_allocated_bytes = 0;
  uint64_t workspace_bytes = 0;
  bool cuda_kv_cache_used = false;
  int steady_state_token_allocations = 0;
};

bool decoder_cuda_generate(const TransformerState& state,
                           const NativeTokenizer& tokenizer,
                           const std::vector<uint16_t>& prompt_tokens,
                           const DecoderSampler& sampler,
                           DecoderKvCache* cache,
                           DecoderCudaGenerateResult* result,
                           std::string* error);

}  // namespace lkjai
