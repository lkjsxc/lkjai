#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "decoder_cuda_block.hpp"
#include "decoder_kv_cache.hpp"
#include "runtime_device.hpp"
#include "transformer_state.hpp"

namespace lkjai {

struct DecoderCudaLayerCacheView {
  DecoderKvCache* cache = nullptr;
  int layer = 0;
  int start_position = 0;
  bool cached_attention = false;
};

class DecoderCudaLayerForward {
 public:
  DecoderCudaLayerForward(const TransformerConfig& cfg,
                          const TransformerLayer& layer,
                          CudaExecutionContext* ctx,
                          DeviceWorkspace* workspace,
                          size_t workspace_bytes);

  void run(const DeviceTensor& x, int batch, int seq, DeviceTensor* out,
           DecoderCudaForwardSubstrateReport* report);
  void run(const DeviceTensor& x, int batch, int seq, DeviceTensor* out,
           DecoderCudaForwardSubstrateReport* report,
           const DecoderCudaLayerCacheView* cache);
  uint64_t workspace_high_water_bytes() const;

 private:
  void allocate_scratch(int rows);
  void upload_projection(DeviceTensor* dst, const Parameter& src, int in,
                         int out);

  TransformerConfig cfg_;
  CudaExecutionContext* ctx_ = nullptr;
  DeviceWorkspace* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
  int scratch_rows_ = 0;
  int kv_width_ = 0;

  DeviceTensor attn_w_;
  DeviceTensor mlp_w_;
  DeviceTensor wq_;
  DeviceTensor wk_;
  DeviceTensor wv_;
  DeviceTensor wo_;
  DeviceTensor wg_;
  DeviceTensor wu_;
  DeviceTensor wd_;

  DeviceTensor norm_;
  DeviceTensor q_;
  DeviceTensor k_;
  DeviceTensor v_;
  DeviceTensor attn_;
  DeviceTensor o_;
  DeviceTensor attn_resid_;
  DeviceTensor mlp_norm_;
  DeviceTensor gate_;
  DeviceTensor up_;
  DeviceTensor swiglu_;
  DeviceTensor down_;
};

}  // namespace lkjai
