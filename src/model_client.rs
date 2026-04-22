use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::config::Config;

#[derive(Clone)]
pub struct ModelClient {
    mode: Mode,
}

#[derive(Clone)]
enum Mode {
    Http(HttpModel),
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
            Mode::Fake(queue) => queue
                .lock()
                .map_err(|_| "fake model lock poisoned".to_string())?
                .pop_front()
                .ok_or_else(|| "fake model has no response queued".to_string()),
        }
    }

    pub async fn status(&self) -> ModelStatus {
        match &self.mode {
            Mode::Http(model) => {
                let reachable = model.health().await;
                ModelStatus {
                    model: model.name.clone(),
                    api_url: model.url.clone(),
                    loaded: true,
                    reachable,
                    message: if reachable {
                        "model server responding".into()
                    } else {
                        "model server unreachable".into()
                    },
                }
            }
            Mode::Fake(_) => ModelStatus {
                model: "fake".into(),
                api_url: "memory://fake".into(),
                loaded: true,
                reachable: true,
                message: "fake model client configured".into(),
            },
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

    async fn health(&self) -> bool {
        let base = self.url.trim_end_matches("/v1/chat/completions");
        let url = format!("{base}/models");
        match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    return false;
                }
                match response.json::<ModelsResponse>().await {
                    Ok(data) => !data.data.is_empty(),
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }
}
