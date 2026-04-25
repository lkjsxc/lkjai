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
    data: Vec<ModelId>,
}

#[derive(Deserialize)]
struct ModelId {
    #[allow(dead_code)]
    id: String,
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
        let health = self.model.health().await;
        let reachable = health.is_ok();
        ModelStatus {
            model: self.model.name.clone(),
            api_url: self.model.url.clone(),
            loaded: true,
            reachable,
            message: health.unwrap_or_else(|error| error),
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

    async fn health(&self) -> Result<String, String> {
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
        Ok("model server responding".into())
    }
}
