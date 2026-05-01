#include "simple_model.hpp"

#include <fstream>
#include <map>
#include <sstream>

namespace lkjai {

namespace {

std::string hex_key(const std::string& key) {
  const char* digits = "0123456789abcdef";
  std::string out;
  for (unsigned char ch : key) {
    out.push_back(digits[ch >> 4]);
    out.push_back(digits[ch & 15]);
  }
  return out;
}

std::string unhex_key(const std::string& hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    unsigned int value = 0;
    std::stringstream stream;
    stream << std::hex << hex.substr(i, 2);
    stream >> value;
    out.push_back(static_cast<char>(value));
  }
  return out;
}

}  // namespace

std::vector<Transition> train_transitions(const std::string& text,
                                          size_t order) {
  std::map<std::string, unsigned int> next;
  for (size_t i = 0; i + order < text.size(); ++i) {
    next[text.substr(i, order)] = static_cast<unsigned char>(text[i + order]);
  }
  std::vector<Transition> transitions;
  for (const auto& item : next) {
    transitions.push_back({item.first, item.second});
  }
  return transitions;
}

bool write_transition_model(const std::filesystem::path& path,
                            const std::vector<Transition>& transitions) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "LKJAI_TRANSITION_V1\n";
  for (const auto& row : transitions) {
    out << hex_key(row.key) << " " << row.c << "\n";
  }
  return true;
}

std::string generate_transition_text(const std::filesystem::path& path,
                                     const std::string& seed, int max_chars) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  std::getline(in, magic);
  if (magic != "LKJAI_TRANSITION_V1") return "";
  std::map<std::string, char> next;
  std::string key;
  unsigned int c = 0;
  while (in >> key >> c) {
    next[unhex_key(key)] = static_cast<char>(c);
  }
  std::string out = seed;
  for (int i = 0; i < max_chars; ++i) {
    if (next.empty() || out.size() < next.begin()->first.size()) break;
    auto order = next.begin()->first.size();
    auto found = next.find(out.substr(out.size() - order, order));
    if (found == next.end()) break;
    out.push_back(found->second);
    if (out.find("</action>") != std::string::npos) break;
  }
  return out;
}

}  // namespace lkjai
