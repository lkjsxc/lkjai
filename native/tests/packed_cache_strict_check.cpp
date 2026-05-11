#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "packed_cache_build.hpp"

namespace {

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

void write_file(const std::filesystem::path& path, const std::string& body) {
  std::ofstream out(path);
  out << body;
}

std::filesystem::path root() {
  auto dir = std::filesystem::temp_directory_path() / "lkjai-packed-strict";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

std::string tokenizer_json() {
  return "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<unk>\":1,\"a\":2,"
         "\"aa\":3},\"merges\":[[\"a\",\"a\"]]},"
         "\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
         "\"added_tokens\":[{\"id\":1,\"content\":\"<unk>\","
         "\"special\":true}]}";
}

bool strict_build_validate() {
  auto dir = root();
  auto source = dir / "source.jsonl";
  auto tokenizer = dir / "tokenizer.json";
  auto config = dir / "config.json";
  auto cache = dir / "cache";
  write_file(source, "{\"text\":\"aaaaaaaaaaaaaaaa\"}\n");
  write_file(tokenizer, tokenizer_json());
  write_file(config, "{\"vocab_size\":16,\"context\":4}\n");
  lkjai::PackedCacheBuildOptions opt;
  opt.source = source;
  opt.tokenizer = tokenizer;
  opt.config = config;
  opt.out = cache;
  opt.seq_len = 4;
  opt.sequence_count = 2;
  std::string error;
  if (!expect(lkjai::build_packed_cache(opt, &error), error)) return false;
  if (!expect(lkjai::validate_packed_cache_command(
                  cache, source, tokenizer, config, false, &error),
              error)) return false;
  write_file(cache / "tokens.bin", "bad");
  return expect(!lkjai::validate_packed_cache_command(
                    cache, source, tokenizer, config, false, &error),
                "corrupt tokens rejected");
}

bool directory_source_is_sorted_and_streamed() {
  auto dir = root();
  auto source = dir / "source";
  std::filesystem::create_directories(source);
  auto tokenizer = dir / "tokenizer.json";
  auto config = dir / "config.json";
  auto cache = dir / "cache";
  write_file(source / "000002.jsonl", "{\"text\":\"aaaaaaaa\"}\n");
  write_file(source / "000001.jsonl", "{\"content\":\"aaaaaaaa\"}\n");
  write_file(tokenizer, tokenizer_json());
  write_file(config, "{\"vocab_size\":16,\"context\":4}\n");
  lkjai::PackedCacheBuildOptions opt;
  opt.source = source;
  opt.tokenizer = tokenizer;
  opt.config = config;
  opt.out = cache;
  opt.seq_len = 4;
  opt.sequence_count = 3;
  std::string error;
  if (!expect(lkjai::build_packed_cache(opt, &error), error)) return false;
  return expect(lkjai::validate_packed_cache_command(
                    cache, source, tokenizer, config, false, &error),
                error);
}

bool missing_tokenizer_rejected() {
  auto dir = root();
  auto source = dir / "source.jsonl";
  auto config = dir / "config.json";
  write_file(source, "{\"text\":\"aaaa\"}\n");
  write_file(config, "{\"vocab_size\":16,\"context\":4}\n");
  lkjai::PackedCacheBuildOptions opt;
  opt.source = source;
  opt.config = config;
  opt.out = dir / "cache";
  opt.seq_len = 4;
  std::string error;
  return expect(!lkjai::build_packed_cache(opt, &error),
                "missing tokenizer rejected");
}

}  // namespace

int main() {
  return strict_build_validate() && directory_source_is_sorted_and_streamed() &&
                 missing_tokenizer_rejected()
             ? 0
             : 1;
}
