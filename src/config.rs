use std::{env, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub tool_timeout_secs: u64,
    pub tool_output_limit: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let data_dir = env_path("DATA_DIR", "/app/data");
        let model_dir = env::var("MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| data_dir.join("models/lkj-150m"));
        Self {
            host: env::var("APP_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            port: env_parse("APP_PORT", 8080),
            data_dir,
            model_dir,
            tool_timeout_secs: env_parse("TOOL_TIMEOUT_SECS", 20),
            tool_output_limit: env_parse("TOOL_OUTPUT_LIMIT", 12_000),
        }
    }

    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn runs_dir(&self) -> PathBuf {
        self.data_dir.join("agent/runs")
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
