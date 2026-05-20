#pragma once

#include <cstdint>

#include <cudnn.h>

#include "runtime_device.hpp"

namespace lkjai {

struct DecoderCudnnSdpaPlanKey {
  int batch = 0;
  int seq = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  bool backward = false;
  bool causal = true;
  int device_id = 0;
  long long cudnn_runtime_version = 0;
};

struct DecoderCudnnSdpaStats {
  bool executed = false;
  bool plan_cache_hit = false;
  bool plan_cache_miss = false;
  uint64_t workspace_bytes = 0;
};

bool decoder_cudnn_sdpa_eligible(const DecoderCudnnSdpaPlanKey& key);

void decoder_cudnn_sdpa_forward_bf16_gqa(
    cudnnHandle_t handle, DeviceWorkspace* workspace, const void* q_bf16,
    const void* k_bf16, const void* v_bf16, void* out_bf16, void* stats_f32,
    const DecoderCudnnSdpaPlanKey& key, DecoderCudnnSdpaStats* stats);

void decoder_cudnn_sdpa_backward_bf16_gqa(
    cudnnHandle_t handle, DeviceWorkspace* workspace, const void* q_bf16,
    const void* k_bf16, const void* v_bf16, const void* out_bf16,
    const void* d_out_bf16, const void* stats_f32, void* d_q_bf16,
    void* d_k_bf16, void* d_v_bf16, const DecoderCudnnSdpaPlanKey& key,
    DecoderCudnnSdpaStats* stats);

}  // namespace lkjai
