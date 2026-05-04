#include "decoder_chat_request.hpp"

#include <cctype>
#include <cmath>
#include <limits>

namespace lkjai {
namespace {

size_t value_pos(std::string_view text, std::string_view key,
                 bool* present, std::string* error) {
  auto needle = "\"" + std::string(key) + "\"";
  auto pos = text.find(needle);
  *present = pos != std::string_view::npos;
  if (!*present) return 0;
  pos = text.find(':', pos + needle.size());
  if (pos == std::string_view::npos) {
    *error = std::string(key) + " missing value";
    return text.size();
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  return pos;
}

bool value_end_ok(std::string_view text, size_t end) {
  while (end < text.size() &&
         std::isspace(static_cast<unsigned char>(text[end]))) {
    ++end;
  }
  return end >= text.size() || text[end] == ',' || text[end] == '}' ||
         text[end] == ']';
}

bool int_field(std::string_view text, std::string_view key, int fallback,
               int* out, std::string* error) {
  bool present = false;
  size_t pos = value_pos(text, key, &present, error);
  if (!present) {
    *out = fallback;
    return true;
  }
  if (pos >= text.size()) return false;
  try {
    size_t consumed = 0;
    long long value = std::stoll(std::string(text.substr(pos)), &consumed);
    if (consumed == 0 || !value_end_ok(text, pos + consumed) ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
      *error = std::string(key) + " must be an integer";
      return false;
    }
    *out = static_cast<int>(value);
    return true;
  } catch (...) {
    *error = std::string(key) + " must be an integer";
    return false;
  }
}

bool float_field(std::string_view text, std::string_view key, float fallback,
                 float* out, std::string* error) {
  bool present = false;
  size_t pos = value_pos(text, key, &present, error);
  if (!present) {
    *out = fallback;
    return true;
  }
  if (pos >= text.size()) return false;
  try {
    size_t consumed = 0;
    float value = std::stof(std::string(text.substr(pos)), &consumed);
    if (consumed == 0 || !std::isfinite(value) ||
        !value_end_ok(text, pos + consumed)) {
      *error = std::string(key) + " must be finite";
      return false;
    }
    *out = value;
    return true;
  } catch (...) {
    *error = std::string(key) + " must be numeric";
    return false;
  }
}

bool seed_field(std::string_view text, uint64_t* out, std::string* error) {
  bool present = false;
  size_t pos = value_pos(text, "seed", &present, error);
  if (!present) return true;
  if (pos >= text.size()) return false;
  if (text[pos] == '-') {
    *error = "seed must be non-negative";
    return false;
  }
  try {
    size_t consumed = 0;
    *out = static_cast<uint64_t>(
        std::stoull(std::string(text.substr(pos)), &consumed));
    if (consumed == 0 || !value_end_ok(text, pos + consumed)) {
      *error = "seed must be an integer";
      return false;
    }
    return true;
  } catch (...) {
    *error = "seed must be an integer";
    return false;
  }
}

}  // namespace

bool parse_decoder_sampler(std::string_view request_body, int vocab_size,
                           DecoderSampler* sampler, std::string* error) {
  if (!int_field(request_body, "max_tokens", 16, &sampler->max_tokens, error)) {
    return false;
  }
  if (sampler->max_tokens < 1 || sampler->max_tokens > 512) {
    *error = "max_tokens must be in [1,512]";
    return false;
  }
  if (!float_field(request_body, "temperature", 0.0f,
                   &sampler->temperature, error)) {
    return false;
  }
  if (sampler->temperature < 0.0f) {
    *error = "temperature must be non-negative";
    return false;
  }
  if (!int_field(request_body, "top_k", 0, &sampler->top_k, error)) {
    return false;
  }
  if (sampler->top_k < 0 || sampler->top_k > vocab_size) {
    *error = "top_k must be in [0,vocab_size]";
    return false;
  }
  if (!float_field(request_body, "top_p", 1.0f, &sampler->top_p, error)) {
    return false;
  }
  if (sampler->top_p <= 0.0f || sampler->top_p > 1.0f) {
    *error = "top_p must be in (0,1]";
    return false;
  }
  return seed_field(request_body, &sampler->seed, error);
}

}  // namespace lkjai
