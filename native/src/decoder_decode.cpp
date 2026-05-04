#include "decoder_decode.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

#include "json_min.hpp"
#include "packed_cache.hpp"
#include "transformer_state.hpp"

namespace lkjai {
namespace {

std::vector<uint16_t> prompt_tokens(const std::string& text, int vocab) {
  std::vector<uint16_t> out;
  for (unsigned char ch : text) out.push_back(static_cast<uint16_t>(ch % vocab));
  if (out.empty()) out.push_back(1);
  return out;
}

int argmax(const std::vector<float>& logits) {
  return static_cast<int>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::string decode_tokens(const std::vector<uint16_t>& tokens) {
  std::ostringstream out;
  for (auto id : tokens) {
    if (id >= 32 && id < 127) {
      out << static_cast<char>(id);
    } else {
      out << "<tok_" << id << ">";
    }
  }
  return out.str();
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
                       std::string* error) {
  TransformerState state;
  if (!load_transformer_artifact(model_dir, &state, error)) return false;
  auto contents = json_string_values(request_body, "content");
  std::string prompt;
  for (const auto& item : contents) prompt += item + "\n";
  auto tokens = prompt_tokens(prompt, state.cfg.vocab_size);
  int prompt_count = static_cast<int>(tokens.size());
  int max_tokens = json_int_value(request_body, "max_tokens", 16);
  max_tokens = std::clamp(max_tokens, 1, 32);
  std::vector<uint16_t> generated;
  for (int i = 0; i < max_tokens; ++i) {
    auto batch = batch_for(tokens, state.cfg.context);
    auto fwd = transformer_forward(batch, state);
    int next = argmax(fwd.next_logits);
    generated.push_back(static_cast<uint16_t>(next));
    tokens.push_back(static_cast<uint16_t>(next));
    if (next == 3) break;
  }
  auto content = decode_tokens(generated);
  int total = prompt_count + static_cast<int>(generated.size());
  *json = "{\"id\":\"chatcmpl-lkjai-decoder\",\"object\":\"chat.completion\","
          "\"model\":\"" + json_escape(model_name) + "\",\"choices\":[{"
          "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
          json_escape(content) + "\"},\"finish_reason\":\"length\"}],"
          "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
          ",\"completion_tokens\":" + std::to_string(generated.size()) +
          ",\"total_tokens\":" + std::to_string(total) + "}}";
  return true;
}

}  // namespace lkjai
