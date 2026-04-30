use lkjai::{agent::ChatRequest, config::Config};
use std::{collections::VecDeque, path::Path, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::Mutex,
};

pub fn request(message: &str, max_steps: usize) -> ChatRequest {
    ChatRequest {
        message: message.into(),
        run_id: Some("run-1".into()),
        max_steps: Some(max_steps),
        visible_event_kinds: None,
    }
}

pub fn action(tool: &str, fields: &[(&str, &str)]) -> String {
    let mut text = format!("<action>\n<reasoning>use tool</reasoning>\n<tool>{tool}</tool>\n");
    for (key, value) in fields {
        text.push_str(&format!("<{key}>{value}</{key}>\n"));
    }
    text.push_str("</action>");
    text
}

pub fn finish(content: &str) -> String {
    action("agent.finish", &[("content", content)])
}

pub fn confirm_action() -> String {
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

pub fn temp_root() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()))
}

pub fn test_config(root: &Path, url: &str) -> Config {
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

pub async fn model_server(responses: Vec<String>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
    let handle = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 65536];
            let size = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..size]);
            let body = if request.starts_with("GET /v1/models") {
                r#"{"data":[{"id":"test-model","object":"model"}]}"#.to_string()
            } else {
                let content = queue
                    .lock()
                    .await
                    .pop_front()
                    .unwrap_or_else(|| finish("empty"));
                format!(
                    r#"{{"choices":[{{"message":{{"role":"assistant","content":{}}}}}]}}"#,
                    serde_json::to_string(&content).unwrap()
                )
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (url, handle)
}
