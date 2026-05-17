#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_decode.hpp"
#include "decoder_kv_cache.hpp"
#include "native_tokenizer.hpp"
#include "transformer_state.hpp"

namespace {

lkjai::NativeTokenizer tokenizer() {
  lkjai::NativeTokenizer out;
  out.vocab_size = 8192;
  out.eos_id = -1;
  return out;
}

bool load_debug_config(lkjai::TransformerConfig* cfg, std::string* error) {
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  return lkjai::load_transformer_config(
      repo / "configs" / "native" / "decoder_debug_bf16.json", cfg, error);
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable\n";
    return 1;
  }
  lkjai::TransformerConfig cfg;
  std::string error;
  if (!load_debug_config(&cfg, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  lkjai::TransformerState state;
  lkjai::init_transformer_state(cfg, &state);
  lkjai::DecoderCudaInferenceSession session(state);
  lkjai::DecoderKvCache cache;
  lkjai::DecoderKvCacheConfig kv_cfg{cfg.layers, 1, cfg.kv_heads, cfg.context,
                                     cfg.head_dim};
  if (!lkjai::decoder_kv_cache_allocate(kv_cfg, &cache, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  lkjai::DecoderSampler sampler;
  sampler.max_tokens = 4;
  sampler.temperature = 0.0f;
  lkjai::DecoderCudaGenerateResult result;
  std::vector<uint16_t> prompt{1, 7, 11, 19};
  if (!session.generate(tokenizer(), prompt, sampler, &cache, &result,
                        &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (result.prefill_allocated_bytes == 0 ||
      result.steady_state_token_allocations != 0 ||
      !result.cuda_kv_cache_used) {
    std::cerr << "decode allocation accounting failed prefill="
              << result.prefill_allocated_bytes << " steady="
              << result.steady_state_token_allocations << " used="
              << result.cuda_kv_cache_used << "\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"decode_backend\":\"cuda_kv_cache\""
            << ",\"prefill_allocated_bytes\":"
            << result.prefill_allocated_bytes
            << ",\"steady_state_token_allocations\":0}\n";
  return 0;
}
