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
        tool_timeout_secs: 20,
        tool_output_limit: 12_000,
    }
}
