#include "runtime_api.hpp"

#include <sstream>

#include "json_min.hpp"
#include "runtime_tool_registry.hpp"

namespace lkjai {

std::string runtime_config_status_json(const RuntimeConfig& cfg) {
  bool local_bind = cfg.host == "127.0.0.1" || cfg.host == "localhost";
  bool has_token = !cfg.kjxlkj_bearer_token.empty();
  std::string resource_base = cfg.kjxlkj_api_url + "/api/users/" +
                              cfg.kjxlkj_user + "/resources";
  std::ostringstream out;
  out << "{\"service\":\"lkjai-native-runtime\""
      << ",\"status\":\"" << (has_token ? "configured" : "degraded") << "\""
      << ",\"degraded\":" << (has_token ? "false" : "true")
      << ",\"degraded_reason\":\""
      << (has_token ? "" : "KJXLKJ_BEARER_TOKEN not configured") << "\""
      << ",\"bind\":{\"host\":\"" << json_escape(cfg.host)
      << "\",\"port\":" << cfg.port
      << ",\"local_only\":" << (local_bind ? "true" : "false") << "}"
      << ",\"data_dir\":\"" << json_escape(cfg.data_dir) << "\""
      << ",\"workspace_dir\":\"" << json_escape(cfg.workspace_dir) << "\""
      << ",\"tool_profile\":\"" << json_escape(cfg.tool_profile) << "\""
      << ",\"kjxlkj\":{\"api_url\":\"" << json_escape(cfg.kjxlkj_api_url)
      << "\",\"user\":\"" << json_escape(cfg.kjxlkj_user)
      << "\",\"bearer_token_configured\":"
      << (has_token ? "true" : "false")
      << ",\"resource_base\":\"" << json_escape(resource_base)
      << "\",\"mutable_tools_enabled\":"
      << (runtime_mutable_tools_enabled(cfg) ? "true" : "false") << "}"
      << ",\"tools\":" << runtime_tool_config_json(cfg) << "}";
  return out.str();
}

}  // namespace lkjai
