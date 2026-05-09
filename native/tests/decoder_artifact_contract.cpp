#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "artifact.hpp"
#include "artifact_manifest.hpp"
#include "decoder_decode.hpp"
#include "json_min.hpp"
#include "train_report.hpp"
#include "transformer_state.hpp"

namespace {

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

std::filesystem::path write_tokenizer(const std::filesystem::path& dir) {
  auto path = dir / "tokenizer.json";
  std::ofstream out(path);
  out << "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"A\":10}},"
      << "\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"added_tokens\":[";
  int id = 100;
  for (const auto& tag : {"<pad>", "<unk>", "<bos>", "<eos>",
                          "<assistant_action>", "<dialogue>", "</dialogue>",
                          "<message>", "</message>", "<role>", "</role>",
                          "<tool_name>", "</tool_name>", "<content>",
                          "</content>", "<action>", "</action>"}) {
    if (id != 100) out << ",";
    out << "{\"id\":" << id++ << ",\"content\":\"" << tag
        << "\",\"special\":true}";
  }
  out << "],\"merges\":[]}\n";
  return path;
}

lkjai::TransformerConfig cfg() {
  lkjai::TransformerConfig out;
  out.model = "decoder-contract";
  out.kind = "decoder";
  out.vocab_size = 512;
  out.context = 8;
  out.layers = 1;
  out.hidden_size = 32;
  out.heads = 4;
  out.kv_heads = 2;
  out.head_dim = 8;
  out.ffn_size = 64;
  out.tie_embeddings = true;
  return out;
}

bool artifact_contract() {
  auto root = std::filesystem::temp_directory_path() / "lkjai-decoder-artifact";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto tokenizer = write_tokenizer(root);
  lkjai::TransformerState state;
  init_transformer_state(cfg(), &state);
  std::string checksum;
  auto export_dir = root / "export";
  auto checkpoint_dir = root / "checkpoint";
  if (!write_transformer_artifact(export_dir, state, 2, 2, 1, 4, 1, 1.0,
                                  false, &checksum, tokenizer)) {
    return expect(false, "failed to write decoder export");
  }
  if (!write_transformer_artifact(checkpoint_dir, state, 2, 2, 1, 4, 1, 1.0,
                                  true, &checksum, tokenizer)) {
    return expect(false, "failed to write decoder checkpoint");
  }
  auto manifest = lkjai::read_text(export_dir / "manifest.json");
  auto tok = lkjai::read_text(export_dir / "tokenizer.json");
  auto index = lkjai::read_text(export_dir / "weights.index.json");
  auto opt_index = lkjai::read_text(checkpoint_dir / "optimizer.index.json");
  std::string error;
  std::string logits_json;
  return expect(lkjai::contains_json_string(manifest, "kind", "decoder"),
                "manifest kind") &&
         expect(lkjai::contains_json_string(
                    manifest, "tokenizer_checksum",
                    lkjai::artifact_text_checksum(tok)),
                "tokenizer checksum") &&
         expect(index.find("pos_embeddings") == std::string::npos,
                "decoder export wrote pos_embeddings") &&
         expect(index.find("\"name\":\"lm_head\"") == std::string::npos,
                "tied decoder export wrote duplicate lm_head") &&
         expect(opt_index.find("pos_embeddings") == std::string::npos,
                "decoder checkpoint wrote pos optimizer") &&
         expect(opt_index.find("master.lm_head") == std::string::npos,
                "tied decoder checkpoint wrote duplicate lm_head optimizer") &&
         expect(lkjai::contains_json_string(
                    manifest, "embedding_tying", "tok_embeddings:lm_head"),
                "manifest embedding tying") &&
         expect(lkjai::inspect_artifact(export_dir, &error), error) &&
         expect(lkjai::transformer_logits_check(export_dir, "1,2,3",
                                                &logits_json, &error),
                error) &&
         expect(lkjai::contains_json_string(logits_json, "status", "pass"),
                "logits check status");
}

bool report_contract() {
  lkjai::TransformerTrainReport r;
  auto root = std::filesystem::temp_directory_path() / "lkjai-decoder-artifact";
  r.model_kind = "decoder";
  r.implementation_status = "partial_cuda";
  r.decoder_status = "partial_cuda";
  r.decoder_cuda_path = true;
  r.forward_backend = "cuda_bf16_embedding_lm_head";
  r.attention_backend = "not_implemented";
  r.decoder_backward_backend = "not_implemented";
  r.kv_cache_backend = lkjai::kDecoderNoKvCacheBackend;
  r.decode_backend = lkjai::kDecoderPartialDecodeBackend;
  r.decode_supported = true;
  r.embedding_tying = "tok_embeddings:lm_head";
  r.trainable_tensor_count = 11;
  r.embedding_weight_changed = true;
  r.lm_head_weight_changed = true;
  r.trainable_weight_changed = true;
  r.logits_check_passed = true;
  r.config_path = root / "export" / "config.json";
  r.checkpoint_dir = root / "checkpoint";
  r.export_dir = root / "export";
  r.served_dir = root / "served";
  lkjai::CudaStatus cuda;
  auto json = lkjai::transformer_train_report_json(r, cuda, "train", "success",
                                                   "");
  return expect(json.find("\"accepted_cuda_training\":false") != std::string::npos,
                "accepted flag") &&
         expect(json.find("\"decoder_block_weight_changed\":false") !=
                    std::string::npos,
                "decoder block weight field") &&
         expect(json.find("\"embedding_weight_changed\":true") !=
                    std::string::npos,
                "embedding weight field") &&
         expect(json.find("\"lm_head_weight_changed\":true") !=
                    std::string::npos,
                "lm head weight field") &&
         expect(json.find("\"non_embedding_weight_changed\":false") !=
                    std::string::npos,
                "partial non embedding weight field") &&
         expect(json.find("\"logits_check_passed\":true") != std::string::npos,
                "logits check passed field") &&
         expect(json.find("\"decoder_forward_probe\"") != std::string::npos,
                "decoder forward probe object") &&
         expect(json.find("\"decoder_block_weights_not_updated\"") !=
                    std::string::npos,
                "decoder block weight limitation") &&
         expect(json.find("\"decoder_block_optimizer_not_implemented\"") !=
                    std::string::npos,
                "decoder block optimizer limitation") &&
         expect(json.find("\"attention_not_implemented\"") != std::string::npos,
                "attention limitation") &&
         expect(json.find("\"decode_backend\":\"host_reference_recompute\"") !=
                    std::string::npos,
                "decode backend") &&
         expect(json.find("\"embedding_tying\":\"tok_embeddings:lm_head\"") !=
                    std::string::npos,
                "embedding tying report");
}

}  // namespace

int main() { return artifact_contract() && report_contract() ? 0 : 1; }
