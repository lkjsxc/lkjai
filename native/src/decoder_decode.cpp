#include "decoder_decode.hpp"

#include "decoder_chat_request.hpp"
#include "decoder_cuda_decode.hpp"
#include "decoder_kv_cache.hpp"
#include "json_min.hpp"
#include "native_tokenizer.hpp"
#include "transformer_report_acceptance.hpp"
#include "transformer_state.hpp"

namespace lkjai {
namespace {

bool accepted_decode_artifact(const std::filesystem::path& model_dir) {
  auto sidecar = read_text(model_dir / "decoder_acceptance.json");
  std::string error;
  return transformer_emitted_decoder_route_report_accepted(
             model_dir / "decoder_train_report.json", &error) &&
         json_bool_value(sidecar, "decode_supported", false) &&
         contains_json_string(sidecar, "decode_backend",
                              kDecoderAcceptedDecodeBackend) &&
         contains_json_string(sidecar, "kv_cache_backend",
                              kDecoderAcceptedKvCacheBackend) &&
         contains_json_string(sidecar, "runtime_path",
                              "accepted_cuda_kv_cache") &&
         json_int_value(sidecar, "kv_cache_steady_state_token_allocations",
                        -1) == 0;
}

bool accepted_state_shape(const TransformerConfig& cfg) {
  return cfg.kind == "decoder" && cfg.context == 1024 && cfg.layers == 10 &&
         cfg.hidden_size == 576 && cfg.heads == 8 && cfg.kv_heads == 2 &&
         cfg.head_dim == 72 && cfg.ffn_size == 1536 && cfg.tie_embeddings;
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
  int prompt_count = static_cast<int>(tokens.size());
  DecoderKvCache cache;
  DecoderKvCacheConfig kv_cfg{state.cfg.layers, 1, state.cfg.kv_heads,
                              state.cfg.context, state.cfg.head_dim};
  if (!decoder_kv_cache_allocate(kv_cfg, &cache, error)) return false;
  DecoderCudaGenerateResult generated;
  if (!decoder_cuda_generate(state, tokenizer, tokens, sampler, &cache,
                             &generated, error)) {
    return false;
  }
  bool accepted_decode =
      accepted_decode_artifact(model_dir) && generated.cuda_kv_cache_used &&
      generated.steady_state_token_allocations == 0 &&
      cache.allocated_bytes > 0 && accepted_state_shape(state.cfg);
  auto content = tokenizer_decode(tokenizer, generated.generated, true);
  int total = prompt_count + static_cast<int>(generated.generated.size());
  *json = "{\"id\":\"chatcmpl-lkjai-decoder\",\"object\":\"chat.completion\","
          "\"model\":\"" + json_escape(model_name) + "\",\"choices\":[{"
          "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
          json_escape(content) + "\"},\"finish_reason\":\"" +
          generated.finish_reason + "\",\"lkjai_stop_reason\":\"" +
          generated.stop_reason + "\",\"lkjai_decode_backend\":\"" +
          std::string(accepted_decode ? kDecoderAcceptedDecodeBackend
                                      : kDecoderRuntimePartialDecodeBackend) +
          "\",\"lkjai_kv_cache_backend\":\"" +
          std::string(accepted_decode ? kDecoderAcceptedKvCacheBackend
                                      : kDecoderRuntimePartialKvCacheBackend) +
          "\",\"lkjai_kv_prefill_allocated_bytes\":" +
          std::to_string(cache.allocated_bytes) +
          ",\"lkjai_kv_steady_state_token_allocations\":" +
          std::to_string(generated.steady_state_token_allocations) + ","
          "\"lkjai_decode_supported\":true"
          ",\"lkjai_decode_accepted\":" +
          std::string(accepted_decode ? "true" : "false") + "}],"
          "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
          ",\"completion_tokens\":" +
          std::to_string(generated.generated.size()) +
          ",\"total_tokens\":" + std::to_string(total) + "}}";
  *http_status = 200;
  return true;
}

}  // namespace lkjai
