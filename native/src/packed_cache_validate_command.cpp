#include "packed_cache_build.hpp"

#include "json_min.hpp"
#include "packed_cache.hpp"
#include "packed_cache_digest.hpp"

namespace lkjai {

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
    if (json_first_string(meta, "source_digest") != packed_source_digest(source) ||
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
