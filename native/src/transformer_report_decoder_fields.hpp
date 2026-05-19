#pragma once

#include <string>

#include "transformer_train.hpp"

namespace lkjai {

std::string transformer_decoder_runtime_report_json_fields(
    const TransformerTrainReport& report);

}  // namespace lkjai
