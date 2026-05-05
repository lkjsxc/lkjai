#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "decoder_cuda_norm.hpp"
#include "decoder_cuda_block_check_ref.hpp"
#include "runtime_device.hpp"
#include "train_report.hpp"

int main() {
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  std::string error;
  lkjai::TransformerConfig debug;
  lkjai::TransformerConfig mid;
  if (!lkjai::load_transformer_config(
          repo / "configs" / "native" / "decoder_debug_bf16.json", &debug,
          &error) ||
      !lkjai::load_transformer_config(
          repo / "configs" / "native" / "decoder_18m_bf16_3070.json", &mid,
          &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  lkjai::DecoderCudaBlockShape shape;
  if (!lkjai::decoder_cuda_block_shape(debug, &shape, &error) ||
      shape.q_width != debug.hidden_size ||
      shape.k_width != debug.kv_heads * debug.head_dim ||
      shape.v_width != debug.kv_heads * debug.head_dim ||
      shape.o_width != debug.hidden_size ||
      shape.ffn_width != debug.ffn_size ||
      shape.gqa_group_size != debug.heads / debug.kv_heads ||
      !lkjai::decoder_cuda_block_shape(mid, &shape, &error)) {
    std::cerr << (error.empty() ? "decoder metadata mismatch" : error) << "\n";
    return 1;
  }

  try {
    lkjai::CudaExecutionContext ctx;
    constexpr int rows = 4;
    constexpr int hidden = 32;
    constexpr float eps = 1.0e-5f;
    std::vector<float> input(static_cast<size_t>(rows) * hidden);
    std::vector<float> weight(hidden);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = std::sin(static_cast<float>(i) * 0.17f) * 1.3f;
    }
    for (int h = 0; h < hidden; ++h) {
      weight[h] = 0.75f + static_cast<float>(h % 11) * 0.03f;
    }
    lkjai::DeviceTensor x({lkjai::DeviceDType::bf16, {rows, hidden}},
                          ctx.stream());
    lkjai::DeviceTensor w({lkjai::DeviceDType::f32, {hidden}},
                          ctx.stream());
    lkjai::DeviceTensor y({lkjai::DeviceDType::bf16, {rows, hidden}},
                          ctx.stream());
    x.copy_from_host_f32(input, ctx.stream());
    w.copy_from_host_f32(weight, ctx.stream());
    lkjai::decoder_launch_rmsnorm_bf16(x.data(),
                                       static_cast<const float*>(w.data()),
                                       y.data(), rows, hidden, eps,
                                       ctx.stream());
    if (!close_enough(y.copy_to_host_f32(ctx.stream()),
                      cpu_rmsnorm(input, weight, rows, hidden, eps), 0.018,
                      0.004, "RMSNorm")) {
      return 1;
    }

    constexpr int batch = 1;
    constexpr int seq = 3;
    constexpr int heads = 2;
    constexpr int head_dim = 8;
    std::vector<float> rope_input(batch * seq * heads * head_dim);
    for (size_t i = 0; i < rope_input.size(); ++i) {
      rope_input[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.7f;
    }
    lkjai::DeviceTensor rope(
        {lkjai::DeviceDType::bf16, {batch, seq, heads, head_dim}},
        ctx.stream());
    rope.copy_from_host_f32(rope_input, ctx.stream());
    lkjai::decoder_launch_rope_bf16(rope.data(), batch, seq, heads, head_dim,
                                    debug.rope_theta, ctx.stream());
    if (!close_enough(rope.copy_to_host_f32(ctx.stream()),
                      cpu_rope(rope_input, batch, seq, heads, head_dim,
                               debug.rope_theta),
                      0.018, 0.004, "RoPE")) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  lkjai::DecoderCudaForwardSubstrateReport probe;
  if (!lkjai::decoder_cuda_forward_substrate_probe(debug, &probe, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (!probe.outputs_finite || !probe.rmsnorm_checked ||
      !probe.rope_checked || !probe.qkv_projection_checked ||
      !probe.attention_checked || !probe.o_projection_checked ||
      !probe.swiglu_checked ||
      probe.shape.q_width != debug.hidden_size ||
      probe.projection_workspace_bytes == 0) {
    std::cerr << "decoder forward substrate probe did not prove all checks\n";
    return 1;
  }

  lkjai::TransformerTrainReport report;
  report.config_path = repo / "configs" / "native" / "decoder_debug_bf16.json";
  report.model_kind = "decoder";
  report.implementation_status = "partial_cuda";
  report.decoder_status = "partial_cuda";
  report.decoder_cuda_path = true;
  report.decoder_cuda_slice = "embedding_lm_head";
  report.decoder_block_backend = "cuda_forward_partial";
  report.rmsnorm_backend = "cuda_bf16_fp32_reduce";
  report.rope_backend = "cuda_bf16";
  report.qkv_projection_backend = "cuda_bf16_cublaslt";
  report.attention_backend = "not_implemented";
  report.mlp_backend = "cuda_swiglu_partial";
  report.decoder_backward_backend = "not_implemented";
  report.kv_cache_backend = "none";
  report.decode_backend = "host_reference_recompute";
  auto json = lkjai::transformer_train_report_json(
      report, lkjai::cuda_status(), "decoder", "success", "");
  if (!require_contains(json, "\"accepted_cuda_training\":false") ||
      !require_contains(json,
                        "\"decoder_block_backend\":\"cuda_forward_partial\"") ||
      !require_contains(json,
                        "\"rmsnorm_backend\":\"cuda_bf16_fp32_reduce\"") ||
      !require_contains(json, "\"rope_backend\":\"cuda_bf16\"") ||
      !require_contains(json,
                        "\"qkv_projection_backend\":\"cuda_bf16_cublaslt\"") ||
      !require_contains(json, "\"attention_backend\":\"not_implemented\"") ||
      !require_contains(json, "\"mlp_backend\":\"cuda_swiglu_partial\"") ||
      !require_contains(json,
                        "\"decoder_backward_backend\":\"not_implemented\"") ||
      !require_contains(json, "\"kv_cache_backend\":\"none\"") ||
      !require_contains(json,
                        "\"decode_backend\":\"host_reference_recompute\"")) {
    return 1;
  }
  report.implementation_status = "accepted";
  report.decoder_status = "accepted";
  report.decoder_cuda_slice = "full_decoder";
  report.decoder_block_backend = "cuda_full_decoder";
  report.forward_backend = "cuda_full_decoder";
  report.backward_backend = "cuda_full_decoder";
  report.attention_backend = "cuda_causal_gqa_bf16_reference";
  report.mlp_backend = "cuda_full_swiglu";
  report.decoder_backward_backend = "cuda_full_decoder";
  report.kv_cache_backend = "cuda_contiguous_bf16";
  report.decode_backend = "cuda_kv_cache";
  json = lkjai::transformer_train_report_json(report, lkjai::cuda_status(),
                                              "decoder", "success", "");
  if (!require_contains(json, "\"accepted_cuda_training\":true") ||
      !require_contains(json, "\"implementation_status\":\"accepted\"") ||
      !require_contains(json, "\"decoder_cuda_slice\":\"full_decoder\"") ||
      !require_contains(json, "\"kv_cache_backend\":\"cuda_contiguous_bf16\"")) {
    return 1;
  }

  std::cout << "{\"status\":\"pass\",\"decoder_block_backend\":"
            << "\"cuda_forward_partial\"}\n";
  return 0;
}
