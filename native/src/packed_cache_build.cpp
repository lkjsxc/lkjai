#include "packed_cache_build.hpp"

#include <algorithm>
#include <fstream>
#include <vector>

#include "json_min.hpp"
#include "native_tokenizer.hpp"
#include "packed_cache.hpp"

namespace lkjai {
namespace {

void write_u16(std::ofstream& out, uint16_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u64(std::ofstream& out, uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::vector<uint16_t> source_tokens(const PackedCacheBuildOptions& opt,
                                    int vocab_size, std::string* error) {
  auto body = read_text(opt.source);
  NativeTokenizer tokenizer;
  bool has_tokenizer = !opt.tokenizer.empty() &&
                       load_native_tokenizer(opt.tokenizer, &tokenizer, error);
  std::vector<uint16_t> tokens;
  for (const auto& key : {"text", "content"}) {
    for (const auto& value : json_string_values(body, key)) {
      auto row = has_tokenizer ? tokenizer_encode(tokenizer, value)
                               : std::vector<uint16_t>{};
      if (row.empty()) {
        for (unsigned char ch : value) row.push_back((ch % (vocab_size - 1)) + 1);
      }
      tokens.insert(tokens.end(), row.begin(), row.end());
    }
  }
  if (tokens.empty()) {
    *error = "source produced no tokens";
  }
  return tokens;
}

}  // namespace

bool build_packed_cache(const PackedCacheBuildOptions& opt, std::string* error) {
  if (opt.seq_len <= 1 || opt.source.empty() || opt.out.empty()) {
    *error = "build requires --source, --out, and --seq-len > 1";
    return false;
  }
  int vocab_size = json_int_value(read_text(opt.config), "vocab_size", 65536);
  if (vocab_size <= 1 || vocab_size > 65536) {
    *error = "config vocab_size must be in [2,65536]";
    return false;
  }
  auto tokens = source_tokens(opt, vocab_size, error);
  if (tokens.empty()) return false;
  int windows = static_cast<int>(tokens.size() / opt.seq_len);
  if (opt.sequence_count > 0) windows = std::min(windows, opt.sequence_count);
  if (windows <= 0) {
    *error = "not enough tokens for one fixed window";
    return false;
  }
  std::filesystem::create_directories(opt.out);
  std::ofstream tok(opt.out / "tokens.bin", std::ios::binary);
  std::ofstream mask(opt.out / "loss_mask.bin", std::ios::binary);
  std::ofstream starts(opt.out / "starts.bin", std::ios::binary);
  for (int w = 0; w < windows; ++w) {
    write_u64(starts, static_cast<uint64_t>(w * opt.seq_len));
    for (int i = 0; i < opt.seq_len; ++i) {
      uint16_t id = tokens[static_cast<size_t>(w * opt.seq_len + i)];
      write_u16(tok, id < vocab_size ? id : 1);
      mask.put(i + 1 == opt.seq_len ? '\0' : '\1');
    }
  }
  std::ofstream meta(opt.out / "metadata.json");
  meta << "{\"format\":\"lkjai-packed-cache-v2\",\"split\":\""
       << json_escape(opt.split) << "\",\"objective\":\""
       << json_escape(opt.objective) << "\",\"sequence_len\":" << opt.seq_len
       << ",\"seq_len\":" << opt.seq_len << ",\"vocab_size\":" << vocab_size
       << ",\"token_dtype\":\"uint16\",\"row_count\":" << windows
       << ",\"sequence_count\":" << windows << ",\"token_count\":"
       << (windows * opt.seq_len) << ",\"seed\":" << opt.seed
       << ",\"run_id\":\"" << json_escape(opt.run_id) << "\"}\n";
  return true;
}

bool validate_packed_cache_command(const std::filesystem::path& cache,
                                   const std::filesystem::path& config,
                                   std::string* error) {
  auto status = inspect_packed_cache(cache);
  if (!status.ok) {
    *error = status.error;
    return false;
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
