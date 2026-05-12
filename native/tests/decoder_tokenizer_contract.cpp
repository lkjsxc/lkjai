#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "native_tokenizer.hpp"
#include "native_tokenizer_build.hpp"
#include "native_xml_tags.hpp"
#include "packed_cache_digest.hpp"

namespace {

std::filesystem::path tokenizer_fixture() {
  auto dir = std::filesystem::temp_directory_path() / "lkjai-tokenizer-contract";
  std::filesystem::create_directories(dir);
  auto path = dir / "tokenizer.json";
  lkjai::NativeTokenizerBuildResult result;
  std::string error;
  if (!lkjai::build_native_tokenizer_json(path, 8192, &result, &error)) {
    std::cerr << error << "\n";
    std::exit(1);
  }
  auto second = dir / "tokenizer-again.json";
  lkjai::NativeTokenizerBuildResult result_again;
  if (!lkjai::build_native_tokenizer_json(second, 8192, &result_again,
                                          &error)) {
    std::cerr << error << "\n";
    std::exit(1);
  }
  if (result.vocab_size > 8192 || result.digest != result_again.digest ||
      result.digest != lkjai::packed_file_digest(path)) {
    std::cerr << "tokenizer builder is not deterministic\n";
    std::exit(1);
  }
  return path;
}

}  // namespace

int main() {
  auto tokenizer_path = tokenizer_fixture();
  lkjai::NativeTokenizer tokenizer;
  std::string error;
  if (!lkjai::load_native_tokenizer(tokenizer_path, &tokenizer, &error) ||
      !lkjai::validate_decoder_tokenizer(tokenizer, 8192, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  for (const auto& tag : lkjai::kNativeXmlTags) {
    auto ids = lkjai::tokenizer_encode(tokenizer, tag.text);
    if (ids.size() != 1) {
      std::cerr << "tag is not atomic: " << tag.text << "\n";
      return 1;
    }
  }
  auto xml_ids = lkjai::tokenizer_encode(tokenizer, "<action><tool>x</tool>");
  if (lkjai::tokenizer_decode(tokenizer, xml_ids, true) !=
      "<action><tool>x</tool>") {
    std::cerr << "xml action tags should survive special-token skipping\n";
    return 1;
  }

  std::string body =
      "{\"messages\":[{\"role\":\"system\",\"content\":\"policy\"},"
      "{\"role\":\"user\",\"content\":\"hello\\n<tag>\"},"
      "{\"role\":\"assistant\",\"content\":\"ok\"},"
      "{\"role\":\"tool\",\"name\":\"search.main\",\"content\":\"result\"}]}";
  std::string prompt;
  if (!lkjai::serialize_chat_prompt(body, &prompt, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  std::string want =
      "<dialogue>\n"
      "<message>\n<role>system</role>\n<content>policy</content>\n</message>\n"
      "<message>\n<role>user</role>\n<content>hello\n<tag></content>\n"
      "</message>\n"
      "<message>\n<role>assistant</role>\n<content>ok</content>\n"
      "</message>\n"
      "<message>\n<role>tool</role>\n<tool_name>search.main</tool_name>\n"
      "<content>result</content>\n</message>\n"
      "</dialogue>\n<assistant_action>\n";
  if (prompt != want) {
    std::cerr << "prompt serialization mismatch\n" << prompt;
    return 1;
  }

  std::string round_trip = "ASCII<dialogue><message><content>x</content>";
  auto ids = lkjai::tokenizer_encode(tokenizer, round_trip);
  auto decoded = lkjai::tokenizer_decode(tokenizer, ids, false);
  if (decoded != round_trip) {
    std::cerr << "tokenizer round trip mismatch\n";
    return 1;
  }
  std::string bad;
  if (lkjai::serialize_chat_prompt(
          "{\"messages\":[{\"role\":\"bad\",\"content\":\"x\"}]}",
          &bad, &error)) {
    std::cerr << "bad role accepted\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\"}\n";
  return 0;
}
