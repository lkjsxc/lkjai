#include "decoder_cudnn_sdpa.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <cuda_runtime.h>
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

struct CachedGraph {
  std::shared_ptr<fe::graph::Graph> graph;
  int64_t workspace_bytes = 0;
};

struct CacheKey {
  bool backward = false;
  int batch = 0, seq = 0, heads = 0, kv_heads = 0, head_dim = 0;
  bool causal = true;
  int device_id = 0;
  long long cudnn_runtime_version = 0;
  bool operator<(const CacheKey& o) const {
    return std::tie(backward, batch, seq, heads, kv_heads, head_dim, causal,
                    device_id, cudnn_runtime_version) <
           std::tie(o.backward, o.batch, o.seq, o.heads, o.kv_heads,
                    o.head_dim, o.causal, o.device_id,
                    o.cudnn_runtime_version);
  }
};

std::map<CacheKey, CachedGraph>& plan_cache() {
  static std::map<CacheKey, CachedGraph> cache;
  return cache;
}

void complete_key(DecoderCudnnSdpaPlanKey* k, bool backward) {
  k->backward = backward;
  cudaGetDevice(&k->device_id);
  k->cudnn_runtime_version = static_cast<long long>(cudnnGetVersion());
}

CachedGraph* cached_graph(cudnnHandle_t handle,
                          const DecoderCudnnSdpaPlanKey& k,
                          std::shared_ptr<fe::graph::Graph> graph,
                          DecoderCudnnSdpaStats* stats) {
  CacheKey key{k.backward, k.batch, k.seq, k.heads, k.kv_heads, k.head_dim,
               k.causal, k.device_id, k.cudnn_runtime_version};
  auto& cache = plan_cache();
  auto found = cache.find(key);
  if (found != cache.end()) {
    if (stats) stats->plan_cache_hit = true;
    return &found->second;
  }
  check(graph->build(handle, {fe::HeurMode_t::A}), "cudnn sdpa build");
  int64_t bytes = 0;
  check(graph->get_workspace_size(bytes), "cudnn sdpa workspace size");
  auto inserted = cache.emplace(key, CachedGraph{graph, bytes});
  if (stats) stats->plan_cache_miss = true;
  return &inserted.first->second;
}

void execute(CachedGraph* graph, cudnnHandle_t handle, DeviceWorkspace* workspace,
             std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp,
             DecoderCudnnSdpaStats* stats) {
  void* ws = workspace->allocate(static_cast<size_t>(graph->workspace_bytes));
  check(graph->graph->execute(handle, vp, ws), "cudnn sdpa execute");
  if (stats) {
    stats->executed = true;
    stats->workspace_bytes = static_cast<uint64_t>(graph->workspace_bytes);
  }
}

}  // namespace

bool decoder_cudnn_sdpa_eligible(const DecoderCudnnSdpaPlanKey& k) {
  return k.batch > 0 && k.seq > 0 && k.heads > 0 && k.kv_heads > 0 &&
         k.head_dim > 0 && k.head_dim % 8 == 0 && k.heads % k.kv_heads == 0 &&
         k.causal;
}

void decoder_cudnn_sdpa_forward_bf16_gqa(
    cudnnHandle_t handle, DeviceWorkspace* workspace, const void* q_bf16,
    const void* k_bf16, const void* v_bf16, void* out_bf16, void* stats_f32,
    const DecoderCudnnSdpaPlanKey& k, DecoderCudnnSdpaStats* stats) {
  if (!decoder_cudnn_sdpa_eligible(k))
    throw std::runtime_error("cudnn sdpa forward ineligible shape");
  auto plan = k;
  complete_key(&plan, false);
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
  auto* cached = cached_graph(handle, plan, g, stats);
  execute(cached, handle, workspace,
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
  auto plan = k;
  complete_key(&plan, true);
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
  auto* cached = cached_graph(handle, plan, g, stats);
  execute(cached, handle, workspace,
          {{Q, const_cast<void*>(q_bf16)}, {K, const_cast<void*>(k_bf16)},
           {V, const_cast<void*>(v_bf16)}, {O, const_cast<void*>(out_bf16)},
           {DO, const_cast<void*>(d_out_bf16)},
           {S, const_cast<void*>(stats_f32)}, {DQ, d_q_bf16}, {DK, d_k_bf16},
           {DV, d_v_bf16}},
          stats);
}

}  // namespace lkjai
