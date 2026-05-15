#pragma once

#include <cuda_runtime.h>

namespace lkjai {

void decoder_cuda_gather_embeddings_bf16(const void* table_bf16,
                                         const void* tokens_u16, void* out_bf16,
                                         int rows, int hidden, int vocab,
                                         cudaStream_t stream);
void decoder_cuda_bf16_to_f32(const void* in_bf16, void* out_f32, int elements,
                              cudaStream_t stream);

}  // namespace lkjai
