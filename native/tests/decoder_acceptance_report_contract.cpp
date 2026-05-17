#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include "decoder_decode.hpp"
#include "train_report.hpp"
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
  r.decoder_cuda_path = true; r.decoder_cuda_slice = "full_decoder";
  r.decoder_block_backend = "cuda_full_decoder"; r.forward_backend = "cuda_full_decoder"; r.backward_backend = "cuda_full_decoder";
  r.decoder_backward_backend = "cuda_full_decoder";
  r.decoder_gradient_source = "cuda_device";
  r.attention_backend = lkjai::kDecoderAcceptedAttentionBackend;
  r.non_embedding_weight_changed = true; r.decoder_block_weight_changed = true; r.trainable_weight_changed = true;
  r.decoder_weight_change.embedding = {0.1, 2, 1};
  r.decoder_weight_change.lm_head = {0.1, 2, 1};
  r.decoder_weight_change.non_embedding = {0.2, 2, 1};
  r.decoder_weight_change.decoder_block = {0.3, 2, 1};
  r.decoder_weight_change.changed_tensors = 4;
  r.logits_check_passed = true;
  r.logits_check_json =
      "{\"status\":\"pass\",\"validation_target\":\"exported_bf16_weights\","
      "\"checksum\":\"abc\"}";
  r.loss = 1.0; r.steps = 1; r.loss_tokens = 1;
  r.seq_len = 1024; r.context = 1024; r.layers = 10; r.hidden_size = 576;
  r.heads = 8; r.kv_heads = 2; r.head_dim = 72; r.ffn_size = 1536;
  r.target_seconds = 7200;
  r.config_path = "configs/native/decoder_40m_bf16_3070.json";
  r.train_config_path = "configs/training/decoder_2h_40m_3070.json";
  r.embedding_tying = "tok_embeddings:lm_head";
  r.kv_cache_backend = lkjai::kDecoderAcceptedKvCacheBackend; r.decode_backend = lkjai::kDecoderAcceptedDecodeBackend; r.decode_supported = true;
  r.kv_cache_prefill_allocated_bytes = 4096;
  r.kv_cache_steady_state_token_allocations = 0;
  return r;
}

bool acceptance_contract() {
  auto r = accepted_report();
  auto untied = r;
  untied.embedding_tying = "none";
  auto partial = r;
  partial.decoder_cuda_slice = "embedding_lm_head"; partial.decoder_backward_backend = "not_implemented";
  partial.kv_cache_backend = "none"; partial.decode_backend = lkjai::kDecoderPartialDecodeBackend; partial.decode_supported = false;
  partial.decoder_block_weight_changed = false; partial.non_embedding_weight_changed = false;
  partial.target_seconds = 7200;
  auto head_only = r;
  head_only.decoder_block_weight_changed = false;
  head_only.decoder_weight_change.decoder_block = {};
  auto no_decode = r;
  no_decode.decode_supported = false;
  auto bad_logits = r;
  bad_logits.logits_check_passed = false;
  auto bad_kv_alloc = r;
  bad_kv_alloc.kv_cache_prefill_allocated_bytes = 0;
  auto bad_kv_steady = r;
  bad_kv_steady.kv_cache_steady_state_token_allocations = 1;
  auto bad_partial_decode_claim = r;
  bad_partial_decode_claim.decode_backend = lkjai::kDecoderPartialDecodeBackend; bad_partial_decode_claim.kv_cache_backend = lkjai::kDecoderPartialKvCacheBackend; bad_partial_decode_claim.decode_supported = true;
  auto bad_loss = r;
  bad_loss.loss = INFINITY;
  auto no_steps = r;
  no_steps.steps = 0;
  auto no_tokens = r;
  no_tokens.loss_tokens = 0;
  auto no_weight = r;
  no_weight.trainable_weight_changed = false;
  auto no_quant = r;
  no_quant.decoder_weight_change.decoder_block = {};
  auto limits = lkjai::transformer_report_limitations(partial, false);
  bool ok = expect(lkjai::transformer_report_accepted_decoder(r),
                   "accepted decoder report");
  for (const auto& item : {
           std::pair{&untied, "untied profile rejected"},
           std::pair{&partial, "partial slice rejected"},
           std::pair{&head_only, "lm-head-only update rejected"},
           std::pair{&no_decode, "missing decode support rejected"},
           std::pair{&bad_logits, "failed logits check rejected"},
           std::pair{&bad_kv_alloc, "missing KV allocation rejected"},
           std::pair{&bad_kv_steady, "steady-state allocation rejected"},
           std::pair{&bad_partial_decode_claim, "partial decode support claim rejected"},
           std::pair{&bad_loss, "non-finite loss rejected"},
           std::pair{&no_steps, "zero steps rejected"},
           std::pair{&no_tokens, "zero loss tokens rejected"},
           std::pair{&no_weight, "missing trainable weight change rejected"},
           std::pair{&no_quant, "missing quantitative block delta rejected"}}) {
    ok = ok && expect(!lkjai::transformer_report_accepted_decoder(*item.first),
                      item.second);
  }
  return ok &&
         expect(std::find(limits.begin(), limits.end(),
                          "decoder_block_weights_not_updated") != limits.end(),
                "block weight limitation") &&
         expect(std::find(limits.begin(), limits.end(),
                          "decoder_block_optimizer_not_implemented") != limits.end(),
                "block optimizer limitation") &&
         expect(std::find(limits.begin(), limits.end(),
                          "autoregressive_decode_unsupported") != limits.end(),
                "partial decode limitation") &&
         expect(!limits.empty(), "partial limitations present");
}

bool reference_attention_contract() {
  auto r = accepted_report();
  r.attention_backend = lkjai::kDecoderReferenceAttentionBackend;
  return expect(!lkjai::transformer_report_accepted_decoder(r),
                "custom reference attention is fallback only");
}

bool emitted_evidence_contract() {
  auto root = std::filesystem::temp_directory_path() / "lkjai-decoder-evidence";
  std::filesystem::remove_all(root);
  for (auto name : {"checkpoint", "export", "served"}) {
    std::filesystem::create_directories(root / name);
    std::ofstream(root / name / "manifest.json")
        << "{\"format\":\"lkjai-native-artifact\",\"weights_checksum\":\"x\"}";
  }
  auto r = accepted_report();
  r.checkpoint_dir = root / "checkpoint";
  r.export_dir = root / "export";
  r.served_dir = root / "served";
  auto report = root / "train-report.json";
  lkjai::CudaStatus cuda;
  cuda.device = "NVIDIA GeForce RTX 3070";
  std::ofstream(report) << lkjai::transformer_train_report_json(
      r, cuda, "train", "success", "");
  std::string error;
  bool ok = expect(lkjai::transformer_emitted_decoder_evidence_accepted(
                       report, &error),
                   error);
  std::filesystem::remove(root / "served" / "manifest.json");
  ok = ok && expect(!lkjai::transformer_emitted_decoder_evidence_accepted(
                        report, &error),
                    "missing served artifact evidence rejected");
  auto no_logits = r;
  no_logits.logits_check_passed = false;
  no_logits.logits_check_json =
      "{\"status\":\"fail\",\"validation_target\":\"exported_bf16_weights\","
      "\"checksum\":\"\"}";
  std::ofstream(report) << lkjai::transformer_train_report_json(
      no_logits, cuda, "train", "success", "");
  ok = ok && expect(!lkjai::transformer_emitted_decoder_evidence_accepted(
                        report, &error),
                    "missing logits pass rejected");
  auto host = accepted_report();
  host.implementation_status = "experimental";
  host.forward_backend = "host_reference";
  host.backward_backend = "host_reference";
  host.decoder_backward_backend = "host_reference"; host.decoder_gradient_source = "host_reference";
  std::ofstream(report) << lkjai::transformer_train_report_json(
      host, cuda, "train", "success", "");
  ok = ok && expect(!lkjai::transformer_emitted_decoder_evidence_accepted(
                        report, &error),
                    "host/reference training rejected");
  auto diag = accepted_report();
  diag.backward_backend = "cuda_diagnostic_synthetic"; diag.decoder_backward_backend = "cuda_diagnostic_synthetic"; diag.decoder_gradient_source = "cuda_device_diagnostic";
  std::ofstream(report) << lkjai::transformer_train_report_json(
      diag, cuda, "train", "success", "");
  ok = ok && expect(!lkjai::transformer_emitted_decoder_evidence_accepted(
                        report, &error), "diagnostic gradients rejected");
  auto route_report = root / "decoder_train_report.json";
  std::ofstream(route_report)
      << lkjai::transformer_train_report_json(r, cuda, "train", "success", "");
  ok = ok && expect(lkjai::transformer_emitted_decoder_route_report_accepted(
      route_report, &error), error);
  auto bad_context = r;
  bad_context.context = 2048;
  std::ofstream(route_report) << lkjai::transformer_train_report_json(
      bad_context, cuda, "train", "success", "");
  ok = ok && expect(!lkjai::transformer_emitted_decoder_route_report_accepted(
      route_report, &error), "bad route context rejected");
  cuda.device = "NVIDIA GeForce RTX 5090";
  std::ofstream(route_report)
      << lkjai::transformer_train_report_json(r, cuda, "train", "success", "");
  ok = ok && expect(!lkjai::transformer_emitted_decoder_route_report_accepted(
      route_report, &error), "non-3070 route report rejected");
  return ok;
}

}  // namespace

int main() {
  return acceptance_contract() && reference_attention_contract() &&
      emitted_evidence_contract() ? 0 : 1;
}
