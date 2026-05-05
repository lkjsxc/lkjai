#include "packed_cache_build.hpp"

#include <algorithm>
#include <fstream>
#include <vector>

#include "json_min.hpp"
#include "native_tokenizer.hpp"
#include "packed_cache.hpp"
#include "packed_cache_digest.hpp"

namespace lkjai {
namespace {

void write_u16(std::ofstream& out, uint16_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u64(std::ofstream& out, uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

struct SourceTokens {
  std::vector<uint16_t> tokens;
  int example_count = 0;
  int max_token_id = 0;
};

SourceTokens source_tokens(const PackedCacheBuildOptions& opt, int vocab_size,
                           std::string* error) {
  SourceTokens out;
  auto body = read_text(opt.source);
  NativeTokenizer tokenizer;
  if (!load_native_tokenizer(opt.tokenizer, &tokenizer, error)) return out;
  if (tokenizer.vocab_size > vocab_size) {
    *error = "tokenizer vocab_size exceeds config vocab_size";
    return out;
  }
  for (const auto& key : {"text", "content"}) {
    for (const auto& value : json_string_values(body, key)) {
      auto row = tokenizer_encode(tokenizer, value);
      for (auto id : row) {
        if (id >= vocab_size) {
          *error = "tokenizer produced token id outside config vocab_size";
          out.tokens.clear();
          return out;
        }
        if (id > out.max_token_id) out.max_token_id = id;
      }
      out.tokens.insert(out.tokens.end(), row.begin(), row.end());
      out.example_count += 1;
    }
  }
  if (out.tokens.empty()) {
    *error = "source produced no tokens";
  }
  return out;
}

}  // namespace

bool build_packed_cache(const PackedCacheBuildOptions& opt, std::string* error) {
  if (opt.seq_len <= 1 || opt.source.empty() || opt.tokenizer.empty() ||
      opt.config.empty() || opt.out.empty()) {
    *error = "build requires --source, --tokenizer, --config, --out, and --seq-len > 1";
    return false;
  }
  int vocab_size = json_int_value(read_text(opt.config), "vocab_size", 65536);
  if (vocab_size <= 1 || vocab_size > 65536) {
    *error = "config vocab_size must be in [2,65536]";
    return false;
  }
  auto source = source_tokens(opt, vocab_size, error);
  if (source.tokens.empty()) return false;
  int windows = static_cast<int>(source.tokens.size() / opt.seq_len);
  if (opt.sequence_count > 0) windows = std::min(windows, opt.sequence_count);
  if (windows <= 0) {
    *error = "not enough tokens for one fixed window";
    return false;
  }
  std::filesystem::create_directories(opt.out);
  int max_written_token_id = 0;
  {
    std::ofstream tok(opt.out / "tokens.bin", std::ios::binary);
    std::ofstream mask(opt.out / "loss_mask.bin", std::ios::binary);
    std::ofstream starts(opt.out / "starts.bin", std::ios::binary);
    for (int w = 0; w < windows; ++w) {
      write_u64(starts, static_cast<uint64_t>(w * opt.seq_len));
      for (int i = 0; i < opt.seq_len; ++i) {
        auto index = static_cast<size_t>(w * opt.seq_len + i);
        auto id = source.tokens[index];
        if (id > max_written_token_id) max_written_token_id = id;
        write_u16(tok, id);
        mask.put(i + 1 == opt.seq_len ? '\0' : '\1');
      }
    }
  }
  std::ofstream meta(opt.out / "metadata.json");
  meta << "{\"format\":\"lkjai-packed-cache-v2\",\"split\":\""
       << json_escape(opt.split) << "\",\"objective\":\""
       << json_escape(opt.objective) << "\",\"sequence_len\":" << opt.seq_len
       << ",\"seq_len\":" << opt.seq_len << ",\"vocab_size\":" << vocab_size
       << ",\"token_dtype\":\"uint16\",\"row_count\":" << windows
       << ",\"sequence_count\":" << windows << ",\"example_count\":"
       << source.example_count << ",\"token_count\":"
       << (windows * opt.seq_len) << ",\"seed\":" << opt.seed
       << ",\"run_id\":\"" << json_escape(opt.run_id) << "\","
       << "\"max_token_id\":" << max_written_token_id << ","
       << "\"tokenizer_digest\":\"" << packed_file_digest(opt.tokenizer) << "\","
       << "\"config_digest\":\"" << packed_file_digest(opt.config) << "\","
       << "\"source_digest\":\"" << packed_file_digest(opt.source) << "\","
       << "\"tokens_checksum\":\"" << packed_file_digest(opt.out / "tokens.bin")
       << "\",\"loss_mask_checksum\":\""
       << packed_file_digest(opt.out / "loss_mask.bin") << "\","
       << "\"starts_checksum\":\"" << packed_file_digest(opt.out / "starts.bin")
       << "\",\"packed_data_checksum\":\""
       << packed_payload_digest(opt.out) << "\"}\n";
  return true;
}

bool validate_packed_cache_command(const std::filesystem::path& cache,
                                   const std::filesystem::path& source,
                                   const std::filesystem::path& tokenizer,
                                   const std::filesystem::path& config,
                                   bool allow_smoke_fixture,
                                   std::string* error) {
  auto status = inspect_packed_cache(cache);
  if (!status.ok) {
    *error = status.error;
    return false;
  }
  if (status.smoke_fixture && !allow_smoke_fixture) {
    *error = "packed cache is an explicit smoke fixture";
    return false;
  }
  auto meta = read_text(cache / "metadata.json");
  if (!status.smoke_fixture) {
    if (source.empty() || tokenizer.empty() || config.empty()) {
      *error = "strict validation requires --source, --tokenizer, and --config";
      return false;
    }
    if (json_first_string(meta, "source_digest") != packed_file_digest(source) ||
        json_first_string(meta, "tokenizer_digest") != packed_file_digest(tokenizer) ||
        json_first_string(meta, "config_digest") != packed_file_digest(config)) {
      *error = "packed cache source/tokenizer/config digest mismatch";
      return false;
    }
  }
  int cfg_vocab = json_int_value(read_text(config), "vocab_size", status.vocab_size);
  int cfg_context = json_int_value(read_text(config), "context", status.sequence_len);
  if (status.vocab_size > cfg_vocab || status.sequence_len > cfg_context) {
    *error = "packed cache exceeds native config vocab or context";
    return false;
  }
  return true;
}

}  // namespace lkjai
