mod agent_support;

use agent_support::{confirm_action, model_server, request, temp_root, test_config};
use lkjai::{agent::Agent, model_client::ModelClient};

#[tokio::test]
async fn confirmation_request_is_disabled_in_readonly_profile() {
    let root = temp_root();
    let (url, server) = model_server(vec![confirm_action()]).await;
    let config = test_config(&root, &url);
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent.chat(request("update release notes", 1)).await;
    assert_eq!(response.stop_reason, "invalid_action");
    assert!(response
        .events
        .iter()
        .any(|event| event.kind == "error" && event.content.contains("readonly")));
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
}
