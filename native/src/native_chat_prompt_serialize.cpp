#include "native_tokenizer.hpp"

#include <sstream>

#include "native_chat_prompt_internal.hpp"

namespace lkjai {

bool serialize_chat_prompt(std::string_view request_body, std::string* prompt,
                           std::string* error) {
  std::vector<ChatPromptMessage> parsed;
  if (!parse_chat_messages_ordered(request_body, &parsed, error)) return false;
  std::ostringstream out;
  out << "<dialogue>\n";
  for (const auto& m : parsed) {
    out << "<message>\n<role>" << m.role << "</role>\n";
    if (!m.tool_name.empty()) {
      out << "<tool_name>" << m.tool_name << "</tool_name>\n";
    }
    out << "<content>" << m.content << "</content>\n</message>\n";
  }
  out << "</dialogue>\n<assistant_action>\n";
  *prompt = out.str();
  return true;
}

}  // namespace lkjai
