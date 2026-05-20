#include "packed_cache_build.hpp"
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
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
  int example_count = 0;
  int max_token_id = 0;
  int windows = 0;
};
std::vector<std::filesystem::path> source_shards(const std::filesystem::path& source,
                                                 std::string* error) {
  std::vector<std::filesystem::path> shards;
  if (std::filesystem::is_regular_file(source)) {
    shards.push_back(source);
    return shards;
  }
  if (!std::filesystem::is_directory(source)) {
    *error = "source must be a JSONL file or a directory of JSONL shards";
    return shards;
  }
  for (const auto& entry : std::filesystem::directory_iterator(source)) {
    if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
      shards.push_back(entry.path());
    }
  }
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) *error = "source directory contains no .jsonl shards";
  return shards;
}
std::string row_text(std::string_view row) {
  auto text = json_first_string(row, "text");
  if (!text.empty()) return text;
  return json_first_string(row, "content");
}
std::pair<std::string, std::string> assistant_target(std::string_view row) {
  auto begin = row.find("<action>");
  auto end = row.find("</action>", begin);
  if (begin == std::string_view::npos || end == std::string_view::npos) {
    return {"", ""};
  }
  end += std::string_view("</action>").size();
  return {std::string(row.substr(0, begin)),
          std::string(row.substr(begin, end - begin))};
}
bool write_streamed_source_tokens(const PackedCacheBuildOptions& opt,
                                  const NativeTokenizer& tokenizer,
                                  int vocab_size, std::ofstream& tok,
                                  std::ofstream& mask,
                                  std::ofstream& starts,
                                  SourceTokens* out,
                                  std::string* error) {
  auto shards = source_shards(opt.source, error);
  if (shards.empty()) return false;
  std::vector<uint16_t> window;
  std::vector<char> loss_window;
  window.reserve(static_cast<size_t>(opt.seq_len));
  loss_window.reserve(static_cast<size_t>(opt.seq_len));
  int64_t emitted_tokens = 0;
  auto append_token = [&](uint16_t id, char loss) {
    window.push_back(id);
    loss_window.push_back(loss);
    if (static_cast<int>(window.size()) != opt.seq_len) return false;
    write_u64(starts, static_cast<uint64_t>(emitted_tokens));
    emitted_tokens += opt.seq_len;
    for (int i = 0; i < opt.seq_len; ++i) {
      write_u16(tok, window[static_cast<size_t>(i)]);
      mask.put(loss_window[static_cast<size_t>(i)]);
    }
    window.clear();
    loss_window.clear();
    out->windows += 1;
    return opt.sequence_count > 0 && out->windows >= opt.sequence_count;
  };
  for (const auto& shard : shards) {
    std::ifstream in(shard);
    if (!in) {
      *error = "failed to open source shard: " + shard.string();
      return false;
    }
    std::string line;
    while (std::getline(in, line)) {
      auto value = row_text(line);
      std::vector<std::pair<std::vector<uint16_t>, char>> parts;
      if (opt.objective == "assistant_masked_sft") {
        auto [prefix, target] = assistant_target(line);
        if (target.empty()) continue;
        parts.push_back({tokenizer_encode(tokenizer, prefix), '\0'});
        parts.push_back({tokenizer_encode(tokenizer, target), '\1'});
      } else {
        if (value.empty()) continue;
        parts.push_back({tokenizer_encode(tokenizer, value), '\1'});
      }
      out->example_count += 1;
      for (const auto& part : parts) {
        for (auto id : part.first) {
          if (id >= vocab_size) {
            *error = "tokenizer produced token id outside config vocab_size";
            return false;
          }
          if (id > out->max_token_id) out->max_token_id = id;
          if (append_token(id, part.second)) return true;
        }
      }
    }
  }
  if (out->windows <= 0) *error = "not enough tokens for one fixed window";
  return out->windows > 0;
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
  NativeTokenizer tokenizer;
  if (!load_native_tokenizer(opt.tokenizer, &tokenizer, error)) return false;
  if (tokenizer.vocab_size > vocab_size) {
    *error = "tokenizer vocab_size exceeds config vocab_size";
    return false;
  }
  std::filesystem::create_directories(opt.out);
  SourceTokens source;
  std::ofstream tok(opt.out / "tokens.bin", std::ios::binary);
  std::ofstream mask(opt.out / "loss_mask.bin", std::ios::binary);
  std::ofstream starts(opt.out / "starts.bin", std::ios::binary);
  if (!tok || !mask || !starts) {
    *error = "failed to open packed cache output files";
    return false;
  }
  if (!write_streamed_source_tokens(opt, tokenizer, vocab_size, tok, mask,
                                    starts, &source, error)) {
    return false;
  }
  tok.close();
  mask.close();
  starts.close();
  std::ofstream meta(opt.out / "metadata.json");
  meta << "{\"format\":\"lkjai-packed-cache\",\"split\":\""
       << json_escape(opt.split) << "\",\"objective\":\""
       << json_escape(opt.objective) << "\",\"sequence_len\":" << opt.seq_len
       << ",\"seq_len\":" << opt.seq_len << ",\"vocab_size\":" << vocab_size
       << ",\"token_dtype\":\"uint16\",\"row_count\":" << source.windows
       << ",\"sequence_count\":" << source.windows << ",\"example_count\":"
       << source.example_count << ",\"token_count\":"
       << (source.windows * opt.seq_len) << ",\"seed\":" << opt.seed
       << ",\"run_id\":\"" << json_escape(opt.run_id) << "\","
       << "\"max_token_id\":" << source.max_token_id << ","
       << "\"tokenizer_digest\":\"" << packed_file_digest(opt.tokenizer) << "\","
       << "\"config_digest\":\"" << packed_file_digest(opt.config) << "\","
       << "\"source_digest\":\"" << packed_source_digest(opt.source) << "\","
       << "\"tokens_checksum\":\"" << packed_file_digest(opt.out / "tokens.bin")
       << "\",\"loss_mask_checksum\":\""
       << packed_file_digest(opt.out / "loss_mask.bin") << "\","
       << "\"starts_checksum\":\"" << packed_file_digest(opt.out / "starts.bin")
       << "\",\"packed_data_checksum\":\""
       << packed_payload_digest(opt.out) << "\"}\n";
  return true;
}
}  // namespace lkjai
