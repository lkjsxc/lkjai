#pragma once

#include <string>
#include <vector>

#include "transformer_train.hpp"

namespace lkjai {

bool transformer_report_accepted_decoder(const TransformerTrainReport& report);
std::vector<std::string> transformer_report_limitations(
    const TransformerTrainReport& report, bool accepted_decoder);

}  // namespace lkjai
