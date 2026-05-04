#pragma once

#include <filesystem>
#include <string>

namespace lkjai {

bool decoder_chat_json(const std::filesystem::path& model_dir,
                       const std::string& model_name,
                       const std::string& request_body,
                       std::string* json,
                       std::string* error);

}  // namespace lkjai
