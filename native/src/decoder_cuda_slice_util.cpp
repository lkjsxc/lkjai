#include "decoder_cuda_slice_internal.hpp"

namespace lkjai {

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
                                      ckpt, &report->logits_checksum);
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
                                           &report->logits_checksum));
}

void decoder_fill_cuda_slice_report(DenseCudaState& cuda,
                                    TransformerTrainReport* r) {
  r->implementation_status = "partial_cuda";
  r->transformer_status = "not_applicable";
  r->decoder_status = "partial_cuda";
  r->decoder_cuda_path = true;
  r->decoder_cuda_slice = "embedding_lm_head";
  r->decoder_block_backend = "static_reference";
  r->forward_backend = "cuda_bf16_embedding_lm_head";
  r->backward_backend = "cuda_bf16_embedding_lm_head";
  r->optimizer_backend = "cuda_adamw_fp32";
  r->attention_backend = "not_implemented";
  r->matmul_backend = "cublaslt";
  r->kv_cache_backend = "none";
  r->cublaslt_workspace_bytes = cuda.cublaslt_workspace_bytes();
  r->workspace_high_water_bytes = cuda.workspace_high_water_bytes();
  r->workspace_reallocations = cuda.workspace_reallocations();
}

}  // namespace lkjai
