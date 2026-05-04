#include "native_chat_prompt_internal.hpp"
#include <cctype>
#include <utility>
namespace lkjai {
namespace {

void ws(std::string_view s, size_t* p) {
  while (*p < s.size() && std::isspace(static_cast<unsigned char>(s[*p]))) ++*p;
}

bool str(std::string_view s, size_t* p, std::string* out) {
  ws(s, p);
  if (*p >= s.size() || s[*p] != '"') return false;
  out->clear();
  bool esc = false;
  for (++*p; *p < s.size(); ++*p) {
    char ch = s[*p];
    if (esc) {
      if (ch == '"' || ch == '\\' || ch == '/') out->push_back(ch);
      else if (ch == 'b') out->push_back('\b');
      else if (ch == 'f') out->push_back('\f');
      else if (ch == 'n') out->push_back('\n');
      else if (ch == 'r') out->push_back('\r');
      else if (ch == 't') out->push_back('\t');
      else return false;
      esc = false;
    } else if (ch == '\\') {
      esc = true;
    } else if (ch == '"') {
      ++*p;
      return true;
    } else {
      out->push_back(ch);
    }
  }
  return false;
}

bool skip(std::string_view s, size_t* p) {
  ws(s, p);
  if (*p >= s.size()) return false;
  if (s[*p] == '"') {
    std::string tmp;
    return str(s, p, &tmp);
  }
  if (s[*p] == '{' || s[*p] == '[') {
    char open = s[*p], close = open == '{' ? '}' : ']';
    int depth = 1;
    for (++*p; *p < s.size() && depth > 0; ++*p) {
      if (s[*p] == '"') {
        std::string tmp;
        if (!str(s, p, &tmp)) return false;
        --*p;
      } else if (s[*p] == open) {
        ++depth;
      } else if (s[*p] == close) {
        --depth;
      }
    }
    return depth == 0;
  }
  while (*p < s.size() && s[*p] != ',' && s[*p] != '}' && s[*p] != ']') ++*p;
  return true;
}

bool role_ok(const std::string& v) {
  return v == "system" || v == "user" || v == "assistant" || v == "tool";
}

bool tool_ok(const std::string& v) {
  if (v.empty() || v.size() > 128) return false;
  for (unsigned char ch : v) {
    if (!std::isalnum(ch) && ch != '_' && ch != '-' && ch != '.' && ch != ':') {
      return false;
    }
  }
  return true;
}

bool msg(std::string_view s, size_t* p, ChatPromptMessage* m,
         std::string* error) {
  ws(s, p);
  if (*p >= s.size() || s[*p] != '{') {
    *error = "messages entries must be JSON objects";
    return false;
  }
  bool has_role = false, has_content = false;
  for (++*p;;) {
    ws(s, p);
    if (*p >= s.size()) {
      *error = "unterminated message object";
      return false;
    }
    if (s[*p] == '}') {
      ++*p;
      break;
    }
    std::string key, val;
    if (!str(s, p, &key)) {
      *error = "message fields must have string keys";
      return false;
    }
    ws(s, p);
    if (*p >= s.size() || s[*p] != ':') {
      *error = "message field missing colon";
      return false;
    }
    ++*p;
    if (key == "role" || key == "content" || key == "name" ||
        key == "tool_name") {
      if (!str(s, p, &val)) {
        *error = "message role, content, and tool name must be strings";
        return false;
      }
      if (key == "role") m->role = val, has_role = true;
      else if (key == "content") m->content = val, has_content = true;
      else if (!val.empty()) m->tool_name = val;
    } else if (!skip(s, p)) {
      *error = "malformed message field";
      return false;
    }
    ws(s, p);
    if (*p < s.size() && s[*p] == ',') ++*p;
    else if (*p >= s.size() || s[*p] != '}') {
      *error = "message object must separate fields with commas";
      return false;
    }
  }
  if (!has_role || !has_content) *error = "each message must include role and content";
  else if (!role_ok(m->role)) *error = "message role must be system, user, assistant, or tool";
  else if (!m->tool_name.empty() && !tool_ok(m->tool_name)) *error = "tool name contains unsupported characters";
  else return true;
  return false;
}

bool messages(std::string_view s, size_t p, std::vector<ChatPromptMessage>* out,
              std::string* error) {
  ws(s, &p);
  if (p >= s.size() || s[p] != '[') {
    *error = "chat request messages must be an array";
    return false;
  }
  for (++p;;) {
    ws(s, &p);
    if (p >= s.size()) {
      *error = "unterminated messages array";
      return false;
    }
    if (s[p] == ']') break;
    ChatPromptMessage item;
    if (!msg(s, &p, &item, error)) return false;
    out->push_back(std::move(item));
    ws(s, &p);
    if (p < s.size() && s[p] == ',') ++p;
    else if (p >= s.size() || s[p] != ']') {
      *error = "messages array must separate entries with commas";
      return false;
    }
  }
  if (out->empty()) *error = "chat request must include at least one message";
  return !out->empty();
}

}  // namespace

bool parse_chat_messages_ordered(std::string_view s,
                                 std::vector<ChatPromptMessage>* out,
                                 std::string* error) {
  size_t p = 0;
  ws(s, &p);
  if (p >= s.size() || s[p] != '{') {
    *error = "chat request body must be a JSON object";
    return false;
  }
  for (++p; p < s.size();) {
    ws(s, &p);
    if (p < s.size() && s[p] == '}') break;
    std::string key;
    if (!str(s, &p, &key)) {
      *error = "chat request fields must have string keys";
      return false;
    }
    ws(s, &p);
    if (p >= s.size() || s[p] != ':') {
      *error = "chat request field missing colon";
      return false;
    }
    if (++p, key == "messages") return messages(s, p, out, error);
    if (!skip(s, &p)) {
      *error = "malformed chat request field";
      return false;
    }
    ws(s, &p);
    if (p < s.size() && s[p] == ',') ++p;
  }
  *error = "chat request must include messages";
  return false;
}

}  // namespace lkjai
