#include "decoder_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

#include "decoder_chat_request.hpp"
#include "decoder_kv_cache.hpp"
#include "json_min.hpp"
#include "native_tokenizer.hpp"
#include "packed_cache.hpp"
#include "transformer_state.hpp"

namespace lkjai {
namespace {

int argmax(const std::vector<float>& logits) {
  return static_cast<int>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

PackedBatch batch_for(const std::vector<uint16_t>& tokens, int context) {
  int n = std::min<int>(context, tokens.size());
  PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = n;
  batch.tokens.assign(tokens.end() - n, tokens.end());
  batch.loss_mask.assign(static_cast<size_t>(n), 1);
  return batch;
}

bool append_kv_token(DecoderKvCache* cache, int token, std::string* error) {
  const auto& c = cache->layout.cfg;
  size_t n = static_cast<size_t>(c.layers * c.kv_heads * c.head_dim);
  std::vector<uint16_t> k(n), v(n);
  for (size_t i = 0; i < n; ++i) {
    k[i] = static_cast<uint16_t>((token + static_cast<int>(i)) & 0xffff);
    v[i] = static_cast<uint16_t>((token * 3 + static_cast<int>(i)) & 0xffff);
  }
  return decoder_kv_cache_append(cache, 0, k, v, error);
}

bool accepted_decode_artifact(const std::filesystem::path& model_dir) {
  (void)model_dir;
  // Sidecar metadata cannot promote host recompute into accepted CUDA decode.
  // This must switch only when generation consumes a real CUDA KV cache.
  return false;
}

}  // namespace

bool decoder_chat_json(const std::filesystem::path& model_dir,
                       const std::string& model_name,
                       const std::string& request_body,
                       std::string* json,
                       int* http_status,
                       std::string* error) {
  *http_status = 500;
  TransformerState state;
  if (!load_transformer_artifact(model_dir, &state, error)) return false;
  NativeTokenizer tokenizer;
  if (!load_native_tokenizer(model_dir / "tokenizer.json", &tokenizer, error)) {
    return false;
  }
  if (!validate_decoder_tokenizer(tokenizer, state.cfg.vocab_size, error)) {
    return false;
  }
  std::string prompt;
  if (!serialize_chat_prompt(request_body, &prompt, error)) {
    *http_status = 400;
    return false;
  }
  DecoderSampler sampler;
  if (!parse_decoder_sampler(request_body, state.cfg.vocab_size, &sampler,
                             error)) {
    *http_status = 400;
    return false;
  }
  auto tokens = tokenizer_encode(tokenizer, prompt);
  bool accepted_decode = accepted_decode_artifact(model_dir);
  int prompt_count = static_cast<int>(tokens.size());
  DecoderKvCache cache;
  DecoderKvCacheConfig kv_cfg{state.cfg.layers, 1, state.cfg.kv_heads,
                              state.cfg.context, state.cfg.head_dim};
  if (!decoder_kv_cache_allocate(kv_cfg, &cache, error)) return false;
  int prefill = std::min(prompt_count, state.cfg.context);
  for (int i = prompt_count - prefill; i < prompt_count; ++i) {
    if (!append_kv_token(&cache, tokens[static_cast<size_t>(i)], error)) {
      return false;
    }
  }
  int eos = tokenizer_id(tokenizer, "<eos>", tokenizer.eos_id);
  int end_action = tokenizer_id(tokenizer, "</action>", -1);
  std::vector<uint16_t> generated;
  std::string finish_reason = "length";
  std::string stop_reason = "max_tokens";
  for (int i = 0; i < sampler.max_tokens; ++i) {
    auto batch = batch_for(tokens, state.cfg.context);
    auto fwd = transformer_forward(batch, state);
    int next = sampler.temperature <= 0.0f
                   ? argmax(fwd.next_logits)
                   : sample_next_token(fwd.next_logits, sampler.temperature,
                                       sampler.top_k, sampler.top_p,
                                       sampler.seed, i);
    generated.push_back(static_cast<uint16_t>(next));
    tokens.push_back(static_cast<uint16_t>(next));
    if (cache.next_position[0] < state.cfg.context &&
        !append_kv_token(&cache, next, error)) {
      return false;
    }
    if (next == eos || next == end_action) {
      finish_reason = "stop";
      stop_reason = next == eos ? "eos" : "end_action";
      break;
    }
  }
  auto content = tokenizer_decode(tokenizer, generated, true);
  int total = prompt_count + static_cast<int>(generated.size());
  *json = "{\"id\":\"chatcmpl-lkjai-decoder\",\"object\":\"chat.completion\","
          "\"model\":\"" + json_escape(model_name) + "\",\"choices\":[{"
          "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
          json_escape(content) + "\"},\"finish_reason\":\"" +
          finish_reason + "\",\"lkjai_stop_reason\":\"" +
          stop_reason + "\",\"lkjai_decode_backend\":\"" +
          std::string(accepted_decode ? kDecoderAcceptedDecodeBackend
                                      : kDecoderPartialDecodeBackend) +
          "\",\"lkjai_kv_cache_backend\":\"" +
          std::string(accepted_decode ? kDecoderAcceptedKvCacheBackend
                                      : kDecoderPartialKvCacheBackend) +
          "\",\"lkjai_kv_prefill_allocated_bytes\":" +
          std::to_string(cache.allocated_bytes) +
          ",\"lkjai_kv_steady_state_token_allocations\":0,"
          "\"lkjai_decode_supported\":" +
          std::string(accepted_decode ? "true" : "false") + "}],"
          "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
          ",\"completion_tokens\":" + std::to_string(generated.size()) +
          ",\"total_tokens\":" + std::to_string(total) + "}}";
  *http_status = 200;
  return true;
}

}  // namespace lkjai
