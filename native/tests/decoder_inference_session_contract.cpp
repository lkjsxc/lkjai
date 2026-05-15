#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_decode.hpp"
#include "decoder_kv_cache.hpp"
#include "native_tokenizer.hpp"
#include "runtime_device.hpp"
#include "transformer_state.hpp"

namespace {

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

lkjai::NativeTokenizer tokenizer() {
  lkjai::NativeTokenizer out;
  out.vocab_size = 8192;
  out.eos_id = 3;
  out.vocab["<eos>"] = 3;
  out.id_to_token[3] = "<eos>";
  out.special_ids[3] = true;
  return out;
}

}  // namespace

int main() {
  auto cuda = lkjai::cuda_status();
  if (!lkjai::cuda_required_ok(cuda)) {
    std::cerr << "CUDA unavailable: "
              << (cuda.error.empty() ? cuda.warning : cuda.error) << "\n";
    return 1;
  }
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  lkjai::TransformerConfig cfg;
  std::string error;
  if (!lkjai::load_transformer_config(
          repo / "configs" / "native" / "decoder_debug_bf16.json", &cfg,
          &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  lkjai::TransformerState state;
  lkjai::init_transformer_state(cfg, &state);

  auto before_session = lkjai::device_allocation_stats();
  lkjai::DecoderCudaInferenceSession session(state);
  auto after_session = lkjai::device_allocation_stats();

  lkjai::DecoderKvCache cache;
  lkjai::DecoderKvCacheConfig kv_cfg{cfg.layers, 1, cfg.kv_heads, cfg.context,
                                     cfg.head_dim};
  bool ok = expect(lkjai::decoder_kv_cache_allocate(kv_cfg, &cache, &error),
                   error);

  lkjai::DecoderSampler sampler;
  sampler.max_tokens = 1;
  sampler.temperature = 0.0f;
  lkjai::DecoderCudaGenerateResult result;
  std::vector<uint16_t> prompt{1, 7, 11, 19};
  ok = ok && expect(session.generate(tokenizer(), prompt, sampler, &cache,
                                     &result, &error),
                    error);
  ok = ok && expect(!result.generated.empty(), "session generated no token");
  ok = ok && expect(result.prefill_allocated_bytes == cache.allocated_bytes,
                    "session did not report KV prefill allocation");
  ok = ok && expect(result.prefill_allocated_bytes > 0,
                    "session prefill allocation was not positive");
  ok = ok && expect(result.cuda_kv_cache_used,
                    "session did not execute CUDA KV-cache path");
  ok = ok && expect(result.steady_state_token_allocations == 0,
                    "one-token session should not report steady allocations");
  ok = ok && expect(result.workspace_bytes > 0,
                    "session did not report workspace usage");
  ok = ok && expect(after_session.allocation_count >
                        before_session.allocation_count,
                    "session construction did not preload device buffers");
  if (!ok) return 1;
  std::cout << "{\"status\":\"pass\",\"decoder_inference_session\":true"
            << ",\"prefill_allocated_bytes\":"
            << result.prefill_allocated_bytes
            << ",\"workspace_bytes\":" << result.workspace_bytes << "}\n";
  return 0;
}
