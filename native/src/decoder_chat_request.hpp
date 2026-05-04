#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace lkjai {

struct DecoderSampler {
  int max_tokens = 16;
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 1.0f;
  uint64_t seed = 0xdec0deull;
};

bool parse_decoder_sampler(std::string_view request_body, int vocab_size,
                           DecoderSampler* sampler, std::string* error);

}  // namespace lkjai
