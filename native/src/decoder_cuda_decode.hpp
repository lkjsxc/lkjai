#pragma once

#include <cstdint>
#include <memory>
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
  std::string sampler_backend = "host_full_vocab";
  bool cuda_kv_cache_used = false;
  int steady_state_token_allocations = 0;
};

class DecoderCudaInferenceSession {
 public:
  explicit DecoderCudaInferenceSession(const TransformerState& state);
  DecoderCudaInferenceSession(const DecoderCudaInferenceSession&) = delete;
  DecoderCudaInferenceSession& operator=(const DecoderCudaInferenceSession&) =
      delete;
  DecoderCudaInferenceSession(DecoderCudaInferenceSession&&) noexcept;
  DecoderCudaInferenceSession& operator=(DecoderCudaInferenceSession&&) noexcept;
  ~DecoderCudaInferenceSession();

  bool logits(const std::vector<uint16_t>& tokens, int start_position,
              bool cached, DecoderKvCache* cache, std::vector<float>* out,
              std::string* error);
  bool generate(const NativeTokenizer& tokenizer,
                const std::vector<uint16_t>& prompt_tokens,
                const DecoderSampler& sampler, DecoderKvCache* cache,
                DecoderCudaGenerateResult* result, std::string* error);

  uint64_t workspace_bytes() const;
  int context_size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool decoder_cuda_generate(const TransformerState& state,
                           const NativeTokenizer& tokenizer,
                           const std::vector<uint16_t>& prompt_tokens,
                           const DecoderSampler& sampler,
                           DecoderKvCache* cache,
                           DecoderCudaGenerateResult* result,
                           std::string* error);

}  // namespace lkjai
