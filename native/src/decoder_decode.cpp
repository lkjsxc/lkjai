#include "decoder_decode.hpp"

#include "decoder_chat_request.hpp"
#include "decoder_cuda_decode.hpp"
#include "decoder_kv_cache.hpp"
#include "json_min.hpp"
#include "native_tokenizer.hpp"
#include "transformer_report_acceptance.hpp"
#include "transformer_state.hpp"

#include <memory>
#include <mutex>

namespace lkjai {
namespace {

struct CachedDecoderArtifact {
  std::filesystem::path model_dir;
  TransformerState state;
  NativeTokenizer tokenizer;
};

bool load_cached_decoder(const std::filesystem::path& model_dir,
                         CachedDecoderArtifact** out, std::string* error) {
  static std::mutex mutex;
  static std::unique_ptr<CachedDecoderArtifact> cached;
  std::lock_guard<std::mutex> lock(mutex);
  if (cached && cached->model_dir == model_dir) {
    *out = cached.get();
    return true;
  }
  auto next = std::make_unique<CachedDecoderArtifact>();
  next->model_dir = model_dir;
  if (!load_transformer_artifact(model_dir, &next->state, error)) return false;
  if (!load_native_tokenizer(model_dir / "tokenizer.json", &next->tokenizer,
                             error)) {
    return false;
  }
  if (!validate_decoder_tokenizer(next->tokenizer, next->state.cfg.vocab_size,
                                  error)) {
    return false;
  }
  cached = std::move(next);
  *out = cached.get();
  return true;
}

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
         json_int_value(sidecar, "kv_cache_prefill_allocated_bytes", 0) > 0 &&
         json_int_value(sidecar, "kv_cache_steady_state_token_allocations",
                        -1) == 0;
}

bool accepted_state_shape(const TransformerConfig& cfg) {
  return cfg.kind == "decoder" && cfg.context == 1024 && cfg.layers == 10 &&
         cfg.hidden_size == 576 && cfg.heads == 8 && cfg.kv_heads == 2 &&
         cfg.head_dim == 72 && cfg.ffn_size == 1536 && cfg.tie_embeddings;
}

}  // namespace

DecoderRouteCapability decoder_route_capability(
    const std::filesystem::path& model_dir, std::string_view artifact_kind) {
  DecoderRouteCapability capability;
  capability.decoder_artifact = artifact_kind == "decoder";
  if (!capability.decoder_artifact) return capability;

  capability.decode_supported = true;
  capability.decode_backend = kDecoderRuntimePartialDecodeBackend;
  capability.kv_cache_backend = kDecoderRuntimePartialKvCacheBackend;
  capability.attention_backend = kDecoderReferenceAttentionBackend;
  capability.acceptance_sidecar_present =
      std::filesystem::is_regular_file(model_dir / "decoder_acceptance.json");
  std::string error;
  capability.train_report_accepted =
      transformer_emitted_decoder_route_report_accepted(
          model_dir / "decoder_train_report.json", &error);
  capability.decode_accepted = accepted_decode_artifact(model_dir);
  if (capability.decode_accepted) {
    capability.decode_backend = kDecoderAcceptedDecodeBackend;
    capability.kv_cache_backend = kDecoderAcceptedKvCacheBackend;
    capability.attention_backend = kDecoderAcceptedAttentionBackend;
    return capability;
  }
  capability.degraded_reason =
      error.empty() ? "accepted decoder route evidence missing" : error;
  return capability;
}

bool decoder_chat_json(const std::filesystem::path& model_dir,
                       const std::string& model_name,
                       const std::string& request_body,
                       std::string* json,
                       int* http_status,
                       std::string* error) {
  *http_status = 500;
  CachedDecoderArtifact* cached = nullptr;
  if (!load_cached_decoder(model_dir, &cached, error)) return false;
  const auto& state = cached->state;
  const auto& tokenizer = cached->tokenizer;
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
      generated.prefill_allocated_bytes > 0 && accepted_state_shape(state.cfg);
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
          std::to_string(generated.prefill_allocated_bytes) +
          ",\"lkjai_kv_steady_state_token_allocations\":" +
          std::to_string(generated.steady_state_token_allocations) +
          ",\"lkjai_decode_cuda_kv_cache_used\":" +
          std::string(generated.cuda_kv_cache_used ? "true" : "false") +
          ",\"lkjai_decode_workspace_bytes\":" +
          std::to_string(generated.workspace_bytes) + ","
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
