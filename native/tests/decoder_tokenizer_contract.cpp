#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "native_tokenizer.hpp"

int main() {
  auto repo = std::filesystem::path(std::getenv("LKJAI_REPO_ROOT")
                                        ? std::getenv("LKJAI_REPO_ROOT")
                                        : ".");
  auto tokenizer_path = repo / "data" / "train" / "tokenizer" / "tokenizer.json";
  lkjai::NativeTokenizer tokenizer;
  std::string error;
  if (!lkjai::load_native_tokenizer(tokenizer_path, &tokenizer, &error) ||
      !lkjai::validate_decoder_tokenizer(tokenizer, 8192, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  for (const auto& tag : {"<dialogue>", "</dialogue>", "<message>",
                          "</message>", "<role>", "</role>", "<tool_name>",
                          "</tool_name>", "<content>", "</content>",
                          "<assistant_action>", "<action>", "</action>",
                          "<eos>"}) {
    auto ids = lkjai::tokenizer_encode(tokenizer, tag);
    if (ids.size() != 1) {
      std::cerr << "tag is not atomic: " << tag << "\n";
      return 1;
    }
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

  std::string round_trip = "ASCII\n<dialogue><message><content>x</content>";
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
