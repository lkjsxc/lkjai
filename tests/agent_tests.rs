mod agent_support;

use agent_support::{
    action, confirm_action, finish, model_server, request, temp_root, test_config,
};
use lkjai::{
    agent::{event, Agent, ChatRequest, TranscriptStore},
    model_client::ModelClient,
};

#[test]
fn transcript_round_trips_events() {
    let root = temp_root();
    let store = TranscriptStore::new(root.clone());
    store
        .append_many("run-1", &[event("user", "hello".into(), None, None)])
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
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "memory_write"));
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
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "confirmation_request"));
    assert!(!response
        .events
        .iter()
        .any(|event| event.kind == "tool_call"));
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
    assert!(agent
        .transcript("run-1")
        .unwrap()
        .iter()
        .any(|event| event.kind == "error"));
    std::fs::remove_dir_all(root).unwrap();
}
