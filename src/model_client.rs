use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use crate::{config::Config, policy_model::PolicyModel};

#[derive(Clone)]
pub struct ModelClient {
    mode: Mode,
}

#[derive(Clone)]
enum Mode {
    Http(HttpModel),
    Policy(PolicyModel),
    Fake(Arc<Mutex<VecDeque<String>>>),
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

impl ModelClient {
    pub fn from_config(config: &Config) -> Self {
        if let Some(path) = config.model_api_url.strip_prefix("policy://") {
            return Self {
                mode: Mode::Policy(PolicyModel::load(path.into())),
            };
        }
        Self {
            mode: Mode::Http(HttpModel {
                client: reqwest::Client::new(),
                url: config.model_api_url.clone(),
                name: config.model_name.clone(),
                max_tokens: config.model_max_new_tokens,
                temperature: config.model_temperature,
            }),
        }
    }

    pub fn fake(responses: Vec<String>) -> Self {
        Self {
            mode: Mode::Fake(Arc::new(Mutex::new(VecDeque::from(responses)))),
        }
    }

    pub async fn chat(&self, messages: &[ModelMessage]) -> Result<String, String> {
        match &self.mode {
            Mode::Http(model) => model.chat(messages).await,
            Mode::Policy(model) => model.chat(messages),
            Mode::Fake(queue) => queue
                .lock()
                .map_err(|_| "fake model lock poisoned".to_string())?
                .pop_front()
                .ok_or_else(|| "fake model has no response queued".to_string()),
        }
    }

    pub fn status(&self) -> ModelStatus {
        match &self.mode {
            Mode::Http(model) => ModelStatus {
                model: model.name.clone(),
                api_url: model.url.clone(),
                loaded: true,
                message: "model client configured".into(),
            },
            Mode::Fake(_) => ModelStatus {
                model: "fake".into(),
                api_url: "memory://fake".into(),
                loaded: true,
                message: "fake model client configured".into(),
            },
            Mode::Policy(model) => model.status(),
        }
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
}
