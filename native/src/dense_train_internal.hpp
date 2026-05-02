#pragma once

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

void init_dense_state(const DenseConfig& cfg, DenseTrainState* state);
double dense_forward_backward(const PackedBatch& batch, DenseTrainState* state);
void dense_adamw(std::vector<float>* weight, std::vector<float>* m,
                 std::vector<float>* v, const std::vector<float>& grad,
                 float lr, int step);
bool write_dense_train_artifact(const std::filesystem::path& dir,
                                const DenseTrainState& state, int step,
                                int microsteps, int batch_size, int seq_len,
                                int grad_accum, double loss,
                                bool checkpoint, std::string* checksum);

}  // namespace lkjai
