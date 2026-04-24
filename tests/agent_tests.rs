use lkjai::{
    agent::{event, Agent, ChatRequest, TranscriptStore},
    config::Config,
    model_client::ModelClient,
};
use std::{collections::VecDeque, path::Path, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::Mutex,
};

#[test]
fn transcript_round_trips_events() {
    let root = temp_root();
    let store = TranscriptStore::new(root.clone());
    store.append_many("run-1", &[event("user", "hello".into(), None, None)])
        .unwrap();
    let loaded = store.read("run-1").unwrap();
    assert_eq!(loaded[0].kind, "user");
    assert_eq!(loaded[0].content, "hello");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn agent_runs_real_tool_then_finishes() {
    let root = temp_root();
    let (url, server) = model_server(vec![
        action("memory.write", &[("content", "user likes concise plans")]),
        finish("Noted."),
    ])
    .await;
    let config = test_config(&root, &url);
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent
        .chat(ChatRequest {
            message: "remember this".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(3),
        })
        .await;
    assert_eq!(response.stop_reason, "finish");
    assert_eq!(response.assistant, "Noted.");
    assert!(response.events.iter().any(|event| event.kind == "memory_write"));
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn shell_runs_inside_tool_workspace() {
    let root = temp_root();
    let workspace = root.join("data/workspace").display().to_string();
    let (url, server) = model_server(vec![
        action("shell.exec", &[("command", "pwd")]),
        finish("done"),
    ])
    .await;
    let config = test_config(&root, &url);
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent.chat(request("where am I?", 2)).await;
    assert_eq!(response.stop_reason, "finish");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "tool_result" && event.content.contains(&workspace)));
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn absolute_file_paths_are_rejected_but_loop_can_finish() {
    let root = temp_root();
    let (url, server) = model_server(vec![
        action("fs.read", &[("path", "/etc/passwd")]),
        finish("blocked"),
    ])
    .await;
    let config = test_config(&root, &url);
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent.chat(request("read passwd", 2)).await;
    assert_eq!(response.stop_reason, "finish");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "tool_result" && event.content.contains("absolute paths")));
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn confirmation_request_stops_without_running_mutation() {
    let root = temp_root();
    let (url, server) = model_server(vec![confirm_action()]).await;
    let config = test_config(&root, &url);
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent.chat(request("update release notes", 2)).await;
    assert_eq!(response.stop_reason, "confirmation_required");
    assert_eq!(response.assistant, "Update release notes?");
    assert!(response.events.iter().any(|event| event.kind == "confirmation_request"));
    assert!(!response.events.iter().any(|event| event.kind == "tool_call"));
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn model_unreachable_turn_is_persisted() {
    let root = temp_root();
    let config = test_config(&root, "http://127.0.0.1:9/v1/chat/completions");
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent.chat(request("hello", 1)).await;
    assert_eq!(response.stop_reason, "model_error");
    assert!(agent.transcript("run-1").unwrap().iter().any(|event| event.kind == "error"));
    std::fs::remove_dir_all(root).unwrap();
}

fn request(message: &str, max_steps: usize) -> ChatRequest {
    ChatRequest {
        message: message.into(),
        run_id: Some("run-1".into()),
        max_steps: Some(max_steps),
    }
}

fn action(tool: &str, fields: &[(&str, &str)]) -> String {
    let mut text = format!("<action>\n<reasoning>use tool</reasoning>\n<tool>{tool}</tool>\n");
    for (key, value) in fields {
        text.push_str(&format!("<{key}>{value}</{key}>\n"));
    }
    text.push_str("</action>");
    text
}

fn finish(content: &str) -> String {
    action("agent.finish", &[("content", content)])
}

fn confirm_action() -> String {
    action(
        "agent.request_confirmation",
        &[
            ("summary", "Update release notes?"),
            ("operation", "resource.update_resource"),
            ("pending_tool", "resource.update_resource"),
            ("ref", "release-notes"),
            ("body", "# Updated"),
        ],
    )
}

fn temp_root() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()))
}

fn test_config(root: &Path, url: &str) -> Config {
    Config {
        host: "127.0.0.1".into(),
        port: 8080,
        data_dir: root.join("data"),
        model_api_url: url.into(),
        model_name: "test-model".into(),
        model_max_new_tokens: 64,
        model_temperature: 0.0,
        agent_max_steps: 6,
        agent_repair_attempts: 1,
        tool_workspace_dir: root.join("data/workspace"),
        tool_timeout_secs: 20,
        tool_output_limit: 12_000,
        kjxlkj_api_url: "http://127.0.0.1:8080".into(),
        kjxlkj_session_cookie: String::new(),
    }
}

async fn model_server(responses: Vec<String>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/chat/completions", listener.local_addr().unwrap());
    let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
    let handle = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 8192];
            let size = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..size]);
            let body = if request.starts_with("GET /v1/models") {
                r#"{"data":[{"id":"test-model","object":"model"}]}"#.to_string()
            } else {
                let content = queue.lock().await.pop_front().unwrap_or_else(|| finish("empty"));
                format!(r#"{{"choices":[{{"message":{{"role":"assistant","content":{}}}}}]}}"#, serde_json::to_string(&content).unwrap())
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (url, handle)
}
