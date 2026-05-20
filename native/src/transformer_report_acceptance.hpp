#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "transformer_train.hpp"

namespace lkjai {

bool transformer_report_accepted_decoder(const TransformerTrainReport& report);
bool transformer_report_shape_accepted_decoder(
    const TransformerTrainReport& report);
bool transformer_emitted_decoder_evidence_accepted(
    const std::filesystem::path& train_report, std::string* error);
bool transformer_emitted_decoder_route_report_accepted(
    const std::filesystem::path& train_report, std::string* error);
bool transformer_route_transcript_accepted(
    const std::filesystem::path& route_transcript,
    const std::filesystem::path& train_report, std::string* error);
std::vector<std::string> transformer_report_limitations(
    const TransformerTrainReport& report, bool accepted_decoder);

}  // namespace lkjai
