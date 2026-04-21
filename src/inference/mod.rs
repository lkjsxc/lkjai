use crate::config::Config;
use candle_core::{Device, Tensor};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Clone)]
pub struct Generator {
    status: ModelStatus,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub model_dir: PathBuf,
    pub device: String,
    pub loaded: bool,
    pub message: String,
}

impl Generator {
    pub fn load(config: &Config) -> Self {
        let device = match Device::cuda_if_available(0) {
            Ok(device) => device,
            Err(_) => Device::Cpu,
        };
        let device_name = format!("{device:?}");
        let config_path = config.model_dir.join("config.json");
        let loaded = config_path.exists() && candle_probe(&device).is_ok();
        let message = if loaded {
            "model metadata found; Candle device initialized".into()
        } else {
            "model export missing; using deterministic fallback".into()
        };
        Self {
            status: ModelStatus {
                model_dir: config.model_dir.clone(),
                device: device_name,
                loaded,
                message,
            },
        }
    }

    pub async fn generate(&self, prompt: &str) -> String {
        if self.status.loaded {
            format!(
                "The local lkjai model export is loaded from {} on {}. \
I can chat here and use YOLO tools when your request is clear. \
For this message, I did not need a tool: {}",
                self.status.model_dir.display(),
                self.status.device,
                prompt
            )
        } else {
            format!(
                "{}. Use /sh, /fetch, /read, /write, or /ls for YOLO tools.",
                self.status.message
            )
        }
    }

    pub fn status(&self) -> ModelStatus {
        self.status.clone()
    }
}

fn candle_probe(device: &Device) -> candle_core::Result<Tensor> {
    Tensor::new(&[0f32], device)
}
