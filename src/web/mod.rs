use crate::{
    agent::{Agent, ChatRequest},
    config::Config,
    model_client::ModelClient,
};
use axum::{
    extract::{Path, State},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use std::{net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
struct AppState {
    agent: Agent,
    model: ModelClient,
}

pub async fn serve(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    tokio::fs::create_dir_all(config.runs_dir()).await?;
    let model = ModelClient::from_config(&config);
    let state = Arc::new(AppState {
        agent: Agent::new(config.clone(), model.clone()),
        model,
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/api/chat", post(chat))
        .route("/api/runs/{id}", get(run))
        .route("/api/model", get(model_status))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let addr: SocketAddr = config.bind_addr().parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

async fn healthz() -> &'static str {
    "ok"
}

async fn model_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.model.status())
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> impl IntoResponse {
    Json(state.agent.chat(request).await)
}

async fn run(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> impl IntoResponse {
    match state.agent.transcript(&id) {
        Ok(events) => Json(events).into_response(),
        Err(_) => (axum::http::StatusCode::NOT_FOUND, "run not found").into_response(),
    }
}
