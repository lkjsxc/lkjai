#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "runtime_api.hpp"

namespace lkjai {

std::string runtime_new_run_id();
std::filesystem::path runtime_run_path(const RuntimeConfig& cfg,
                                       const std::string& id);
void runtime_append_event(const RuntimeConfig& cfg, const std::string& run_id,
                          const std::string& kind,
                          const std::string& content, int step = 0,
                          const std::string& tool = "");
std::string runtime_events_json(const RuntimeConfig& cfg,
                                const std::string& run_id,
                                const std::vector<std::string>& visible);
std::string runtime_runs_json(const RuntimeConfig& cfg, int limit);

}  // namespace lkjai
