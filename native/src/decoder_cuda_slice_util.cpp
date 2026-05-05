#include "decoder_cuda_slice_internal.hpp"

#include <sstream>

namespace lkjai {

bool decoder_validate_layer_shapes(const TransformerConfig& cfg,
                                   std::string* error) {
  if (cfg.kind != "decoder") {
    *error = "decoder shape validation requires model_kind=decoder";
    return false;
  }
  if (cfg.hidden_size <= 0 || cfg.heads <= 0 || cfg.kv_heads <= 0 ||
      cfg.head_dim <= 0 || cfg.ffn_size <= 0 || cfg.layers <= 0 ||
      cfg.context <= 1 || cfg.vocab_size <= 0) {
    *error = "decoder config has invalid non-positive dimensions";
    return false;
  }
  if (cfg.heads * cfg.head_dim != cfg.hidden_size) {
    *error = "decoder heads * head_dim must equal hidden_size";
    return false;
  }
  if (cfg.heads % cfg.kv_heads != 0) {
    *error = "decoder heads must be divisible by kv_heads";
    return false;
  }
  if (cfg.ffn_size < cfg.hidden_size) {
    *error = "decoder ffn_size must be at least hidden_size";
    return false;
  }
  return true;
}

std::string decoder_shape_report(const TransformerConfig& cfg) {
  std::ostringstream out;
  int kv = cfg.kv_heads * cfg.head_dim;
  out << "{\"tok_embeddings\":[" << cfg.vocab_size << "," << cfg.hidden_size
      << "],\"attn_norm\":[" << cfg.hidden_size << "],"
      << "\"q_proj\":[" << cfg.hidden_size << "," << cfg.hidden_size
      << "],\"k_proj\":[" << cfg.hidden_size << "," << kv
      << "],\"v_proj\":[" << cfg.hidden_size << "," << kv
      << "],\"o_proj\":[" << cfg.hidden_size << "," << cfg.hidden_size
      << "],\"mlp_norm\":[" << cfg.hidden_size << "],"
      << "\"gate_proj\":[" << cfg.hidden_size << "," << cfg.ffn_size
      << "],\"up_proj\":[" << cfg.hidden_size << "," << cfg.ffn_size
      << "],\"down_proj\":[" << cfg.ffn_size << "," << cfg.hidden_size
      << "],\"final_norm\":[" << cfg.hidden_size << "],"
      << "\"lm_head\":[" << cfg.vocab_size << "," << cfg.hidden_size
      << "]}";
  return out.str();
}

DenseConfig decoder_dense_cfg(const TransformerConfig& cfg) {
  DenseConfig out;
  out.model = cfg.model + "-decoder-cuda-slice";
  out.vocab_size = cfg.vocab_size;
  out.context = cfg.context;
  out.layers = 1;
  out.hidden_size = cfg.hidden_size;
  out.heads = cfg.heads;
  out.kv_heads = cfg.kv_heads;
  out.head_dim = cfg.head_dim;
  out.ffn_size = cfg.ffn_size;
  out.seed = cfg.seed;
  return out;
}

DenseTrainState decoder_dense_state(const DenseConfig& cfg,
                                    const TransformerState& source) {
  DenseTrainState out;
  out.cfg = cfg;
  out.emb = source.tok_embeddings.w;
  out.head = source.lm_head.w;
  out.grad_emb.assign(out.emb.size(), 0.0f);
  out.grad_head.assign(out.head.size(), 0.0f);
  out.m_emb = source.tok_embeddings.m;
  out.v_emb = source.tok_embeddings.v;
  out.m_head = source.lm_head.m;
  out.v_head = source.lm_head.v;
  return out;
}

void decoder_copy_dense_back(const DenseTrainState& dense,
                             TransformerState* state) {
  if (state->cfg.tie_embeddings) {
    state->tok_embeddings.w = dense.emb;
    state->tok_embeddings.m = dense.m_emb;
    state->tok_embeddings.v = dense.v_emb;
    for (size_t i = 0; i < state->tok_embeddings.w.size(); ++i) {
      state->tok_embeddings.w[i] = 0.5f * (dense.emb[i] + dense.head[i]);
      state->tok_embeddings.m[i] = dense.m_emb[i] + dense.m_head[i];
      state->tok_embeddings.v[i] = dense.v_emb[i] + dense.v_head[i];
    }
    state->lm_head.w = state->tok_embeddings.w;
    state->lm_head.m = state->tok_embeddings.m;
    state->lm_head.v = state->tok_embeddings.v;
    return;
  }
  state->tok_embeddings.w = dense.emb;
  state->tok_embeddings.m = dense.m_emb;
  state->tok_embeddings.v = dense.v_emb;
  state->lm_head.w = dense.head;
  state->lm_head.m = dense.m_head;
  state->lm_head.v = dense.v_head;
}

bool decoder_write_all(const TransformerTrainOptions& opt,
                       const TransformerState& state,
                       TransformerTrainReport* report, int seq_len) {
  auto write = [&](const std::filesystem::path& dir, bool ckpt) {
    return write_transformer_artifact(dir, state, report->steps,
                                      report->microsteps, opt.batch_size,
                                      seq_len, opt.grad_accum, report->loss,
                                      ckpt, &report->logits_checksum,
                                      opt.tokenizer_path);
  };
  return write(opt.out_dir / "checkpoints" / "latest", true) &&
         write(opt.out_dir / "checkpoints" / "final", true) &&
         write(report->export_dir, false) && write(report->served_dir, false) &&
         (opt.export_artifact.empty()
              ? true
              : write_transformer_artifact(opt.export_artifact, state,
                                           report->steps, report->microsteps,
                                           opt.batch_size, seq_len,
                                           opt.grad_accum, report->loss, false,
                                           &report->logits_checksum,
                                           opt.tokenizer_path));
}

void decoder_fill_cuda_slice_report(DenseCudaState& cuda,
                                    TransformerTrainReport* r) {
  r->implementation_status = "partial_cuda";
  r->transformer_status = "not_applicable";
  r->decoder_status = "partial_cuda";
  r->decoder_cuda_path = true;
  r->decoder_cuda_slice = "embedding_lm_head";
  r->decoder_block_backend = "cuda_forward_partial";
  r->forward_backend = "cuda_bf16_embedding_lm_head";
  r->backward_backend = "cuda_bf16_embedding_lm_head";
  r->optimizer_backend = "cuda_adamw_fp32";
  r->rmsnorm_backend = "cuda_bf16_fp32_reduce";
  r->rope_backend = "cuda_bf16";
  r->qkv_projection_backend = "cuda_bf16_cublaslt";
  r->attention_backend = "cuda_causal_gqa_bf16_reference";
  r->mlp_backend = "cuda_swiglu_partial";
  r->decoder_backward_backend = "not_implemented";
  r->matmul_backend = "cublaslt";
  r->kv_cache_backend = "none";
  r->decode_backend = "host_reference_recompute";
  r->decode_supported = true;
  r->cublaslt_workspace_bytes = cuda.cublaslt_workspace_bytes();
  r->workspace_high_water_bytes = cuda.workspace_high_water_bytes();
  r->workspace_reallocations = cuda.workspace_reallocations();
}

}  // namespace lkjai
