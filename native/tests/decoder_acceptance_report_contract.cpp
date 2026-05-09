#include <iostream>
#include <string>

#include "decoder_decode.hpp"
#include "transformer_report_acceptance.hpp"

namespace {

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

lkjai::TransformerTrainReport accepted_report() {
  lkjai::TransformerTrainReport r;
  r.model_kind = "decoder";
  r.implementation_status = "accepted";
  r.decoder_cuda_path = true;
  r.decoder_cuda_slice = "full_decoder";
  r.decoder_block_backend = "cuda_full_decoder";
  r.forward_backend = "cuda_full_decoder";
  r.backward_backend = "cuda_full_decoder";
  r.decoder_backward_backend = "cuda_full_decoder";
  r.attention_backend = "cuda_causal_gqa_bf16_reference";
  r.embedding_tying = "tok_embeddings:lm_head";
  r.kv_cache_backend = lkjai::kDecoderAcceptedKvCacheBackend;
  r.decode_backend = lkjai::kDecoderAcceptedDecodeBackend;
  r.decode_supported = true;
  return r;
}

bool acceptance_contract() {
  auto r = accepted_report();
  auto untied = r;
  untied.embedding_tying = "none";
  auto partial = r;
  partial.decoder_cuda_slice = "embedding_lm_head";
  partial.decoder_backward_backend = "not_implemented";
  partial.kv_cache_backend = "none";
  partial.decode_backend = lkjai::kDecoderPartialDecodeBackend;
  auto limits = lkjai::transformer_report_limitations(partial, false);
  return expect(lkjai::transformer_report_accepted_decoder(r),
                "accepted decoder report") &&
         expect(!lkjai::transformer_report_accepted_decoder(untied),
                "untied profile rejected") &&
         expect(!lkjai::transformer_report_accepted_decoder(partial),
                "partial slice rejected") &&
         expect(!limits.empty(), "partial limitations present");
}

bool cudnn_attention_contract() {
  auto r = accepted_report();
  r.attention_backend = "cudnn_sdpa";
  return expect(lkjai::transformer_report_accepted_decoder(r),
                "cudnn accepted attention");
}

}  // namespace

int main() {
  return acceptance_contract() && cudnn_attention_contract() ? 0 : 1;
}
