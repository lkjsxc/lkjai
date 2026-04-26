use std::{env, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub model_api_url: String,
    pub model_name: String,
    pub model_max_new_tokens: usize,
    pub model_temperature: f32,
    pub agent_max_steps: usize,
    pub agent_repair_attempts: usize,
    pub tool_workspace_dir: PathBuf,
    pub tool_timeout_secs: u64,
    pub tool_output_limit: usize,
    pub kjxlkj_api_url: String,
    pub kjxlkj_session_cookie: String,
}

impl Config {
    pub fn from_env() -> Self {
        let data_dir = env_path("DATA_DIR", "/app/data");
        Self {
            host: env::var("APP_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            port: env_parse("APP_PORT", 8080),
            data_dir,
            model_api_url: env::var("MODEL_API_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8081/v1/chat/completions".into()),
            model_name: env::var("MODEL_NAME").unwrap_or_else(|_| "lkjai-scratch-20m".into()),
            model_max_new_tokens: env_parse("MODEL_MAX_NEW_TOKENS", 512),
            model_temperature: env_parse("MODEL_TEMPERATURE", 0.2),
            agent_max_steps: env_parse("AGENT_MAX_STEPS", 6),
            agent_repair_attempts: env_parse("AGENT_REPAIR_ATTEMPTS", 1),
            tool_workspace_dir: env_path("TOOL_WORKSPACE_DIR", "/app/data/workspace"),
            tool_timeout_secs: env_parse("TOOL_TIMEOUT_SECS", 20),
            tool_output_limit: env_parse("TOOL_OUTPUT_LIMIT", 12_000),
            kjxlkj_api_url: env::var("KJXLKJ_API_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080".into()),
            kjxlkj_session_cookie: env::var("KJXLKJ_SESSION_COOKIE").unwrap_or_default(),
        }
    }

    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn runs_dir(&self) -> PathBuf {
        self.data_dir.join("agent/runs")
    }

    pub fn memory_path(&self) -> PathBuf {
        self.data_dir.join("agent/memory.sqlite3")
    }
}

fn env_path(key: &str, default: &str) -> PathBuf {
    env::var(key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| default.into())
}

fn env_parse<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
