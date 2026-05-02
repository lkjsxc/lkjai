#include "dense_cuda_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "json_min.hpp"

namespace lkjai {

double dense_seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

float dense_step_lr(const DenseTrainOptions& opt, int step) {
  if (opt.warmup_steps <= 0 || step > opt.warmup_steps) return opt.lr;
  return opt.lr * static_cast<float>(step) / static_cast<float>(opt.warmup_steps);
}

uint16_t dense_pack_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

float dense_round_bf16(float value) {
  uint32_t bits = static_cast<uint32_t>(dense_pack_bf16(value)) << 16;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string dense_checksum_floats(const std::vector<float>& values) {
  uint64_t hash = 1469598103934665603ull;
  for (float value : values) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 4; ++i) {
      hash = (hash ^ ((bits >> (i * 8)) & 0xffu)) * 1099511628211ull;
    }
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

int dense_resume_step(const std::filesystem::path& dir) {
  if (dir.empty()) return 0;
  return json_int_value(read_text(dir / "trainer_state.json"),
                        "optimizer_steps", 0);
}

int dense_supervised_count(const PackedBatch& batch) {
  int seen = 0;
  for (int row = 0; row < batch.batch_size; ++row) {
    for (int pos = 0; pos + 1 < batch.sequence_len; ++pos) {
      auto base = static_cast<size_t>(row * batch.sequence_len + pos);
      if (batch.loss_mask[base + 1] != 0) ++seen;
    }
  }
  return seen;
}

CpuDenseForward cpu_dense_forward_backward_with_logits(const PackedBatch& batch,
                                                       DenseTrainState* state) {
  CpuDenseForward out;
  out.loss = dense_forward_backward(batch, state);
  const auto& cfg = state->cfg;
  out.logits.assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
  int base = (batch.batch_size - 1) * batch.sequence_len + batch.sequence_len - 2;
  int token = batch.tokens[static_cast<size_t>(base)] % cfg.vocab_size;
  auto* h = state->emb.data() + static_cast<size_t>(token) * cfg.hidden_size;
  for (int v = 0; v < cfg.vocab_size; ++v) {
    auto* w = state->head.data() + static_cast<size_t>(v) * cfg.hidden_size;
    for (int i = 0; i < cfg.hidden_size; ++i) out.logits[v] += h[i] * w[i];
  }
  return out;
}

double dense_max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
  double diff = 0.0;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    diff = std::max(diff, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return diff;
}

}  // namespace lkjai
