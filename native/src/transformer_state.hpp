#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "packed_cache.hpp"
#include "transformer_train.hpp"

namespace lkjai {

struct Parameter {
  std::string name;
  std::vector<int> shape;
  std::vector<float> w;
  std::vector<float> g;
  std::vector<float> m;
  std::vector<float> v;
};

struct TransformerLayer {
  Parameter attn_norm;
  Parameter q_proj;
  Parameter k_proj;
  Parameter v_proj;
  Parameter o_proj;
  Parameter mlp_norm;
  Parameter gate_proj;
  Parameter up_proj;
  Parameter down_proj;
};

struct TransformerState {
  TransformerConfig cfg;
  Parameter tok_embeddings;
  Parameter pos_embeddings;
  std::vector<TransformerLayer> layers;
  Parameter final_norm;
  Parameter lm_head;
};

struct ForwardResult {
  double loss = 0.0;
  std::vector<float> next_logits;
  std::vector<float> last_hidden;
  std::vector<float> loss_logits;
  std::vector<float> loss_hidden;
  int loss_label = 0;
  int supervised = 0;
};

void init_transformer_state(const TransformerConfig& cfg,
                            TransformerState* state);
ForwardResult transformer_forward(const PackedBatch& batch,
                                  const TransformerState& state);
void transformer_backward(const PackedBatch& batch, const ForwardResult& fwd,
                          TransformerState* state);
void transformer_adamw(TransformerState* state, float lr, int step);
bool transformer_cuda_step_probe(std::string* error);
long long transformer_parameter_count(const TransformerState& state);
bool write_transformer_artifact(const std::filesystem::path& dir,
                                const TransformerState& state, int step,
                                int microsteps, int batch_size, int seq_len,
                                int grad_accum, double loss, bool checkpoint,
                                std::string* checksum,
                                const std::filesystem::path& tokenizer_path = {});
bool load_transformer_artifact(const std::filesystem::path& dir,
                               TransformerState* state, std::string* error);
bool load_transformer_checkpoint(const std::filesystem::path& dir,
                                 const TransformerConfig& requested,
                                 int batch_size, int seq_len, int grad_accum,
                                 TransformerState* state, int* optimizer_steps,
                                 int* microsteps, std::string* error);
std::string checksum_logits(const std::vector<float>& logits);

}  // namespace lkjai
