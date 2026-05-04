#include "decoder_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

#include "decoder_chat_request.hpp"
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
          stop_reason + "\"}],"
          "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
          ",\"completion_tokens\":" + std::to_string(generated.size()) +
          ",\"total_tokens\":" + std::to_string(total) + "}}";
  *http_status = 200;
  return true;
}

}  // namespace lkjai
