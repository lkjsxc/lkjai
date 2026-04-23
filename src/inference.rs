use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
struct InferenceState {
    artifact: Arc<ArtifactState>,
}

#[derive(Debug)]
struct ArtifactState {
    model: String,
    root: PathBuf,
    loaded: bool,
    missing: Vec<String>,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    #[allow(dead_code)]
    max_tokens: Option<usize>,
    #[allow(dead_code)]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
}

#[derive(Serialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

pub async fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let host = env::var("INFERENCE_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = env_parse("INFERENCE_PORT", 8081);
    let model = env::var("MODEL_NAME").unwrap_or_else(|_| "lkjai-scratch-40m".into());
    let root = env::var("MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/models"));
    let artifact = Arc::new(ArtifactState::load(model, root));
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .layer(TraceLayer::new_for_http())
        .with_state(InferenceState { artifact });
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn models(State(state): State<InferenceState>) -> impl IntoResponse {
    if !state.artifact.loaded {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: format!(
                    "missing artifact files: {}",
                    state.artifact.missing.join(", ")
                ),
            }),
        )
            .into_response();
    }
    Json(ModelsResponse {
        data: vec![ModelInfo {
            id: state.artifact.model.clone(),
            object: "model",
        }],
    })
    .into_response()
}

async fn chat(
    State(state): State<InferenceState>,
    Json(body): Json<ChatRequest>,
) -> impl IntoResponse {
    if !state.artifact.loaded {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: format!(
                    "model artifact not loaded at {}",
                    state.artifact.root.display()
                ),
            }),
        )
            .into_response();
    }
    let answer = deterministic_action(&body.messages);
    Json(ChatResponse {
        choices: vec![Choice {
            message: AssistantMessage {
                role: "assistant",
                content: answer,
            },
        }],
    })
    .into_response()
}

fn deterministic_action(messages: &[ChatMessage]) -> String {
    let latest = messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .unwrap_or("");
    serde_json::json!({
        "kind": "final",
        "thought": "scratch inference stub",
        "content": format!("scratch inference stub loaded; real decoding is pending. Last user input: {latest}")
    })
    .to_string()
}

impl ArtifactState {
    fn load(model: String, root: PathBuf) -> Self {
        let model_root = root.join(&model);
        let required = ["manifest.json", "config.json", "tokenizer.json", "model.pt"];
        let missing = required
            .iter()
            .filter(|name| !model_root.join(name).is_file())
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        Self {
            model,
            root: model_root,
            loaded: missing.is_empty(),
            missing,
        }
    }
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
