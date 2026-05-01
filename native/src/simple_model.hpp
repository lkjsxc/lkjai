#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace lkjai {

struct Transition {
  std::string key;
  unsigned int c;
};

std::vector<Transition> train_transitions(const std::string& text,
                                          size_t order = 16);
bool write_transition_model(const std::filesystem::path& path,
                            const std::vector<Transition>& transitions);
std::string generate_transition_text(const std::filesystem::path& path,
                                     const std::string& seed, int max_chars);

}  // namespace lkjai
