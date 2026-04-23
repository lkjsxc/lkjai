use lkjai::{
    agent::{event, Agent, ChatRequest, TranscriptStore},
    config::Config,
    model_client::ModelClient,
};

#[test]
fn transcript_round_trips_events() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let store = TranscriptStore::new(root.clone());
    let events = vec![event("user", "hello".into(), None, None)];
    store.append_many("run-1", &events).unwrap();
    let loaded = store.read("run-1").unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].kind, "user");
    assert_eq!(loaded[0].content, "hello");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn agent_runs_tool_then_final() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let config = test_config(&root);
    let model = ModelClient::fake(vec![
        r#"{"kind":"tool_call","thought":"remember","tool":"memory.write","args":{"content":"user likes concise plans"}}"#.into(),
        r#"{"kind":"final","thought":"done","content":"Noted."}"#.into(),
    ]);
    let agent = Agent::new(config, model);
    let response = agent
        .chat(ChatRequest {
            message: "remember this".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(3),
        })
        .await;
    assert_eq!(response.stop_reason, "final");
    assert_eq!(response.assistant, "Noted.");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "memory_write"));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn shell_runs_inside_tool_workspace() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let config = test_config(&root);
    let workspace = config.tool_workspace_dir.display().to_string();
    let model = ModelClient::fake(vec![
        r#"{"kind":"tool_call","thought":"inspect","tool":"shell.exec","args":{"command":"pwd"}}"#
            .into(),
        r#"{"kind":"final","thought":"done","content":"done"}"#.into(),
    ]);
    let agent = Agent::new(config, model);
    let response = agent
        .chat(ChatRequest {
            message: "where am I?".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(2),
        })
        .await;
    assert_eq!(response.stop_reason, "final");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "tool_result" && event.content.contains(&workspace)));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn absolute_file_paths_are_rejected() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let config = test_config(&root);
    let model = ModelClient::fake(vec![
        r#"{"kind":"tool_call","thought":"read","tool":"fs.read","args":{"path":"/etc/passwd"}}"#
            .into(),
        r#"{"kind":"final","thought":"done","content":"blocked"}"#.into(),
    ]);
    let agent = Agent::new(config, model);
    let response = agent
        .chat(ChatRequest {
            message: "read passwd".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(2),
        })
        .await;
    assert_eq!(response.stop_reason, "final");
    assert!(response
        .events
        .iter()
        .any(|event| { event.kind == "tool_result" && event.content.contains("absolute paths") }));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn list_workspace_root_exists_on_new_agent() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let config = test_config(&root);
    let model = ModelClient::fake(vec![
        r#"{"kind":"tool_call","thought":"list","tool":"fs.list","args":{"path":"."}}"#.into(),
        r#"{"kind":"final","thought":"done","content":"done"}"#.into(),
    ]);
    let agent = Agent::new(config, model);
    let response = agent
        .chat(ChatRequest {
            message: "list files".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(2),
        })
        .await;
    assert_eq!(response.stop_reason, "final");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "tool_result" && !event.content.contains("tool failed")));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn model_unreachable_turn_is_persisted() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let mut config = test_config(&root);
    config.model_api_url = "http://127.0.0.1:9/v1/chat/completions".into();
    let model = ModelClient::from_config(&config);
    let agent = Agent::new(config, model);
    let response = agent
        .chat(ChatRequest {
            message: "hello".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(1),
        })
        .await;
    assert_eq!(response.stop_reason, "model_error");
    let loaded = agent.transcript("run-1").unwrap();
    assert!(loaded.iter().any(|event| event.kind == "error"));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn confirmation_request_stops_without_running_mutation() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let config = test_config(&root);
    let model = ModelClient::fake(vec![
        r##"{"kind":"request_confirmation","summary":"Update release notes?","operation":"resource.update_resource","pending_tool_call":{"tool":"resource.update_resource","args":{"ref":"release-notes","body":"# Updated","is_private":false}}}"##.into(),
    ]);
    let agent = Agent::new(config, model);
    let response = agent
        .chat(ChatRequest {
            message: "update release notes".into(),
            run_id: Some("run-1".into()),
            max_steps: Some(2),
        })
        .await;
    assert_eq!(response.stop_reason, "confirmation_required");
    assert_eq!(response.assistant, "Update release notes?");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "confirmation_request"));
    assert!(!response
        .events
        .iter()
        .any(|event| event.kind == "tool_call"));
    std::fs::remove_dir_all(root).unwrap();
}

fn test_config(root: &std::path::Path) -> Config {
    Config {
        host: "127.0.0.1".into(),
        port: 8080,
        data_dir: root.join("data"),
        model_api_url: "memory://fake".into(),
        model_name: "fake".into(),
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
