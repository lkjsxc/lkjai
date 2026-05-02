#pragma once

#include <string>
#include <string_view>

namespace lkjai {

bool validate_manifest(std::string_view text, std::string_view config,
                       std::string_view tokenizer, std::string* kind,
                       std::string* error);

}  // namespace lkjai
