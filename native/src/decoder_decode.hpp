#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

inline constexpr const char* kDecoderPartialDecodeBackend =
    "host_reference_recompute";
inline constexpr const char* kDecoderPartialKvCacheBackend =
    "host_contiguous_bf16_diagnostic";
inline constexpr const char* kDecoderNoKvCacheBackend = "none";
inline constexpr const char* kDecoderAcceptedDecodeBackend = "cuda_kv_cache";
inline constexpr const char* kDecoderAcceptedKvCacheBackend =
    "cuda_contiguous_bf16";

bool decoder_chat_json(const std::filesystem::path& model_dir,
                       const std::string& model_name,
                       const std::string& request_body,
                       std::string* json,
                       int* http_status,
                       std::string* error);

}  // namespace lkjai
