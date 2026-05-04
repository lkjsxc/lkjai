#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace lkjai {

struct ChatPromptMessage {
  std::string role;
  std::string content;
  std::string tool_name;
};

bool parse_chat_messages_ordered(std::string_view request_body,
                                 std::vector<ChatPromptMessage>* out,
                                 std::string* error);

}  // namespace lkjai
