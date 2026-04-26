use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::Config;

#[derive(Clone)]
pub struct ModelClient {
    model: HttpModel,
}

#[derive(Clone)]
struct HttpModel {
    client: reqwest::Client,
    url: String,
    name: String,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub model: String,
    pub api_url: String,
    pub loaded: bool,
    pub reachable: bool,
    pub message: String,
    pub device: String,
    pub cuda_available: bool,
    pub gpu_name: String,
    pub warning: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ModelMessage],
    max_tokens: usize,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: String,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<serde_json::Value>,
    #[serde(default)]
    device: String,
    #[serde(default)]
    cuda_available: bool,
    #[serde(default)]
    gpu_name: String,
    #[serde(default)]
    warning: String,
}

impl ModelClient {
    pub fn from_config(config: &Config) -> Self {
        Self {
            model: HttpModel {
                client: reqwest::Client::new(),
                url: config.model_api_url.clone(),
                name: config.model_name.clone(),
                max_tokens: config.model_max_new_tokens,
                temperature: config.model_temperature,
            },
        }
    }

    pub async fn chat(&self, messages: &[ModelMessage]) -> Result<String, String> {
        self.model.chat(messages).await
    }

    pub async fn status(&self) -> ModelStatus {
        let (reachable, health) = match self.model.health().await {
            Ok(health) => (true, health),
            Err(error) => (false, HealthStatus::error(error)),
        };
        ModelStatus {
            model: self.model.name.clone(),
            api_url: self.model.url.clone(),
            loaded: true,
            reachable,
            message: health.message,
            device: health.device,
            cuda_available: health.cuda_available,
            gpu_name: health.gpu_name,
            warning: health.warning,
        }
    }

    pub async fn is_reachable(&self) -> bool {
        self.status().await.reachable
    }
}

impl HttpModel {
    async fn chat(&self, messages: &[ModelMessage]) -> Result<String, String> {
        let body = ChatRequest {
            model: &self.name,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };
        let response = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("model request failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(format!("model server returned {status}: {text}"));
        }
        let data: ChatResponse = response
            .json()
            .await
            .map_err(|error| format!("model response parse failed: {error}"))?;
        data.choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| "model response had no choices".into())
    }

    async fn health(&self) -> Result<HealthStatus, String> {
        let base = self
            .url
            .strip_suffix("/chat/completions")
            .unwrap_or(self.url.as_str());
        let url = format!("{base}/models");
        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|error| format!("model health request failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(format!("model health returned {status}: {text}"));
        }
        let data = response
            .json::<ModelsResponse>()
            .await
            .map_err(|error| format!("model health parse failed: {error}"))?;
        if data.data.is_empty() {
            return Err("model health returned no models".into());
        }
        Ok(HealthStatus {
            message: "model server responding".into(),
            device: data.device,
            cuda_available: data.cuda_available,
            gpu_name: data.gpu_name,
            warning: data.warning,
        })
    }
}

struct HealthStatus {
    message: String,
    device: String,
    cuda_available: bool,
    gpu_name: String,
    warning: String,
}

impl HealthStatus {
    fn error(message: String) -> Self {
        Self {
            message,
            device: String::new(),
            cuda_available: false,
            gpu_name: String::new(),
            warning: String::new(),
        }
    }
}
