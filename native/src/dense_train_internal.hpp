#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "dense_train.hpp"
#include "packed_cache.hpp"

namespace lkjai {

struct DenseTrainState {
  DenseConfig cfg;
  std::vector<float> emb;
  std::vector<float> head;
  std::vector<float> grad_emb;
  std::vector<float> grad_head;
  std::vector<float> m_emb;
  std::vector<float> v_emb;
  std::vector<float> m_head;
  std::vector<float> v_head;
};

struct DenseCheckpointMetadata {
  int optimizer_steps = 0;
  int microsteps = 0;
  int batch_size = 0;
  int seq_len = 0;
  int grad_accum = 0;
  double loss = 0.0;
  std::string logits_checksum;
};

void init_dense_state(const DenseConfig& cfg, DenseTrainState* state);
uint16_t dense_pack_bf16(float value);
float dense_round_bf16(float value);
double dense_forward_backward(const PackedBatch& batch, DenseTrainState* state);
void dense_adamw(std::vector<float>* weight, std::vector<float>* m,
                 std::vector<float>* v, const std::vector<float>& grad,
                 float lr, int step);
bool load_dense_checkpoint(const std::filesystem::path& dir,
                           const DenseConfig& requested, int batch_size,
                           int seq_len, int grad_accum,
                           DenseTrainState* state,
                           DenseCheckpointMetadata* metadata,
                           std::string* error);
bool write_dense_train_artifact(const std::filesystem::path& dir,
                                const DenseTrainState& state, int step,
                                int microsteps, int batch_size, int seq_len,
                                int grad_accum, double loss,
                                bool checkpoint, std::string* checksum);
bool write_dense_train_artifact_staged(const std::filesystem::path& dir,
                                       const DenseTrainState& state, int step,
                                       int microsteps, int batch_size,
                                       int seq_len, int grad_accum,
                                       double loss, bool checkpoint,
                                       std::string* checksum);
bool write_dense_train_outputs(const DenseTrainOptions& opt,
                               const DenseTrainState& final_state,
                               const DenseTrainState& best_state,
                               int seq_len, DenseTrainReport* report,
                               std::string* error);

}  // namespace lkjai
