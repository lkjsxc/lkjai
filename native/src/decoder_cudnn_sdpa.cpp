#include "decoder_cudnn_sdpa.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cudnn_frontend.h>

namespace lkjai {
namespace {
namespace fe = cudnn_frontend;

constexpr int64_t Q = 1, K = 2, V = 3, O = 4, S = 5;
constexpr int64_t DO = 6, DQ = 7, DK = 8, DV = 9;

void check(fe::error_object status, const char* label) {
  if (status.is_good()) return;
  throw std::runtime_error(std::string(label) + ": " + status.get_message());
}

int64_t q_stride_b(const DecoderCudnnSdpaPlanKey& k) {
  return int64_t(k.seq) * k.heads * k.head_dim;
}

int64_t kv_stride_b(const DecoderCudnnSdpaPlanKey& k) {
  return int64_t(k.seq) * k.kv_heads * k.head_dim;
}

std::shared_ptr<fe::graph::Graph> base_graph() {
  auto g = std::make_shared<fe::graph::Graph>();
  g->set_io_data_type(fe::DataType_t::BFLOAT16)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  return g;
}

auto tensor(std::shared_ptr<fe::graph::Graph> g, const char* name, int64_t uid,
            int64_t b, int64_t h, int64_t s, int64_t d, int64_t sb) {
  return g->tensor(fe::graph::Tensor_attributes()
                       .set_name(name)
                       .set_uid(uid)
                       .set_dim({b, h, s, d})
                       .set_stride({sb, d, h * d, 1}));
}

void execute(std::shared_ptr<fe::graph::Graph> g, cudnnHandle_t handle,
             DeviceWorkspace* workspace,
             std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp,
             DecoderCudnnSdpaStats* stats) {
  check(g->build(handle, {fe::HeurMode_t::A}), "cudnn sdpa build");
  int64_t bytes = 0;
  check(g->get_workspace_size(bytes), "cudnn sdpa workspace size");
  void* ws = workspace->allocate(static_cast<size_t>(bytes));
  check(g->execute(handle, vp, ws), "cudnn sdpa execute");
  if (stats) {
    stats->executed = true;
    stats->workspace_bytes = static_cast<uint64_t>(bytes);
  }
}

}  // namespace

bool decoder_cudnn_sdpa_eligible(const DecoderCudnnSdpaPlanKey& k) {
  return k.batch > 0 && k.seq > 0 && k.heads > 0 && k.kv_heads > 0 &&
         k.head_dim > 0 && k.heads % k.kv_heads == 0;
}

void decoder_cudnn_sdpa_forward_bf16_gqa(
    cudnnHandle_t handle, DeviceWorkspace* workspace, const void* q_bf16,
    const void* k_bf16, const void* v_bf16, void* out_bf16, void* stats_f32,
    const DecoderCudnnSdpaPlanKey& k, DecoderCudnnSdpaStats* stats) {
  if (!decoder_cudnn_sdpa_eligible(k))
    throw std::runtime_error("cudnn sdpa forward ineligible shape");
  auto g = base_graph();
  auto q = tensor(g, "Q", Q, k.batch, k.heads, k.seq, k.head_dim, q_stride_b(k));
  auto key =
      tensor(g, "K", K, k.batch, k.kv_heads, k.seq, k.head_dim, kv_stride_b(k));
  auto val =
      tensor(g, "V", V, k.batch, k.kv_heads, k.seq, k.head_dim, kv_stride_b(k));
  auto opt = fe::graph::SDPA_attributes()
                 .set_name("decoder_sdpa_fwd")
                 .set_generate_stats(true)
                 .set_attn_scale(1.0f / std::sqrt(float(k.head_dim)))
                 .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                 .set_diagonal_band_right_bound(0);
  auto [o, s] = g->sdpa(q, key, val, opt);
  o->set_output(true).set_uid(O).set_dim({k.batch, k.heads, k.seq, k.head_dim})
      .set_stride({q_stride_b(k), k.head_dim, k.heads * k.head_dim, 1});
  s->set_output(true).set_uid(S).set_data_type(fe::DataType_t::FLOAT);
  execute(g, handle, workspace,
          {{Q, const_cast<void*>(q_bf16)}, {K, const_cast<void*>(k_bf16)},
           {V, const_cast<void*>(v_bf16)}, {O, out_bf16}, {S, stats_f32}},
          stats);
}

void decoder_cudnn_sdpa_backward_bf16_gqa(
    cudnnHandle_t handle, DeviceWorkspace* workspace, const void* q_bf16,
    const void* k_bf16, const void* v_bf16, const void* out_bf16,
    const void* d_out_bf16, const void* stats_f32, void* d_q_bf16,
    void* d_k_bf16, void* d_v_bf16, const DecoderCudnnSdpaPlanKey& k,
    DecoderCudnnSdpaStats* stats) {
  if (!decoder_cudnn_sdpa_eligible(k))
    throw std::runtime_error("cudnn sdpa backward ineligible shape");
  auto g = base_graph();
  auto q = tensor(g, "Q", Q, k.batch, k.heads, k.seq, k.head_dim, q_stride_b(k));
  auto key =
      tensor(g, "K", K, k.batch, k.kv_heads, k.seq, k.head_dim, kv_stride_b(k));
  auto val =
      tensor(g, "V", V, k.batch, k.kv_heads, k.seq, k.head_dim, kv_stride_b(k));
  auto o = tensor(g, "O", O, k.batch, k.heads, k.seq, k.head_dim, q_stride_b(k));
  auto dy =
      tensor(g, "dO", DO, k.batch, k.heads, k.seq, k.head_dim, q_stride_b(k));
  auto softmax_stats = g->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Stats")
                                     .set_uid(S)
                                     .set_dim({k.batch, k.heads, k.seq, 1})
                                     .set_stride({k.heads * k.seq, k.seq, 1, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
  auto opt = fe::graph::SDPA_backward_attributes()
                 .set_name("decoder_sdpa_bwd")
                 .set_attn_scale(1.0f / std::sqrt(float(k.head_dim)))
                 .set_deterministic_algorithm(true)
                 .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                 .set_diagonal_band_right_bound(0);
  auto [dq, dk, dv] = g->sdpa_backward(q, key, val, o, dy, softmax_stats, opt);
  dq->set_output(true).set_uid(DQ).set_dim({k.batch, k.heads, k.seq, k.head_dim})
      .set_stride({q_stride_b(k), k.head_dim, k.heads * k.head_dim, 1});
  dk->set_output(true).set_uid(DK).set_dim({k.batch, k.kv_heads, k.seq, k.head_dim})
      .set_stride({kv_stride_b(k), k.head_dim, k.kv_heads * k.head_dim, 1});
  dv->set_output(true).set_uid(DV).set_dim({k.batch, k.kv_heads, k.seq, k.head_dim})
      .set_stride({kv_stride_b(k), k.head_dim, k.kv_heads * k.head_dim, 1});
  execute(g, handle, workspace,
          {{Q, const_cast<void*>(q_bf16)}, {K, const_cast<void*>(k_bf16)},
           {V, const_cast<void*>(v_bf16)}, {O, const_cast<void*>(out_bf16)},
           {DO, const_cast<void*>(d_out_bf16)},
           {S, const_cast<void*>(stats_f32)}, {DQ, d_q_bf16}, {DK, d_k_bf16},
           {DV, d_v_bf16}},
          stats);
}

}  // namespace lkjai
