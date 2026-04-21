mod model;

use crate::config::Config;
use model::LkjModel;
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct Generator {
    status: ModelStatus,
    runtime: Option<Arc<Mutex<Runtime>>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub model_dir: PathBuf,
    pub device: String,
    pub loaded: bool,
    pub message: String,
}

struct Runtime {
    tokenizer: Tokenizer,
    model: LkjModel,
}

impl Generator {
    pub fn load(config: &Config) -> Self {
        match Runtime::load(config) {
            Ok(runtime) => {
                let status = runtime.status(config.model_dir.clone());
                Self {
                    status,
                    runtime: Some(Arc::new(Mutex::new(runtime))),
                }
            }
            Err(error) => Self {
                status: ModelStatus {
                    model_dir: config.model_dir.clone(),
                    device: "unavailable".into(),
                    loaded: false,
                    message: format!("model load failed: {error}"),
                },
                runtime: None,
            },
        }
    }

    pub async fn generate(&self, prompt: &str) -> String {
        let Some(runtime) = self.runtime.clone() else {
            return self.status.message.clone();
        };
        let prompt = prompt.to_owned();
        match tokio::task::spawn_blocking(move || runtime.lock().unwrap().generate(&prompt)).await {
            Ok(Ok(text)) => text,
            Ok(Err(error)) => format!("model generation failed: {error}"),
            Err(error) => format!("model generation task failed: {error}"),
        }
    }

    pub fn status(&self) -> ModelStatus {
        self.status.clone()
    }
}

impl Runtime {
    fn load(config: &Config) -> Result<Self, String> {
        let tokenizer_path = tokenizer_path(config)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        let model = LkjModel::load(&config.model_dir).map_err(|e| e.to_string())?;
        Ok(Self { tokenizer, model })
    }

    fn status(&self, model_dir: PathBuf) -> ModelStatus {
        ModelStatus {
            model_dir,
            device: self.model.device_name(),
            loaded: true,
            message: "model, tokenizer, and weights loaded".into(),
        }
    }

    fn generate(&mut self, prompt: &str) -> Result<String, String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| e.to_string())?;
        let mut ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            return Err("tokenizer produced no input ids".into());
        }
        let prompt_len = ids.len();
        for _ in 0..32 {
            let input = tail(&ids, self.model.context());
            let next = self.model.next_token(input).map_err(|e| e.to_string())?;
            if next == 1 {
                break;
            }
            ids.push(next);
        }
        let generated = ids[prompt_len..].to_vec();
        if generated.is_empty() {
            return Ok(String::new());
        }
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| e.to_string())
    }
}

fn tokenizer_path(config: &Config) -> Result<PathBuf, String> {
    let local = config.model_dir.join("tokenizer.json");
    if local.exists() {
        return Ok(local);
    }
    let shared = config.data_dir.join("tokenizers/tokenizer.json");
    if shared.exists() {
        return Ok(shared);
    }
    Err(format!(
        "tokenizer missing: expected {} or {}",
        display(&local),
        display(&shared)
    ))
}

fn tail(ids: &[u32], limit: usize) -> &[u32] {
    let start = ids.len().saturating_sub(limit);
    &ids[start..]
}

fn display(path: &Path) -> String {
    path.display().to_string()
}
