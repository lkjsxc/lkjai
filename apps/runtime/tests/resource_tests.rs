mod agent_support;

use agent_support::{
    action, confirm_action, model_server, request, resource_server, temp_root, test_config,
};
use lkjai::{agent::Agent, model_client::ModelClient};

#[tokio::test]
async fn direct_resource_mutation_is_blocked_without_confirmation() {
    let root = temp_root();
    let (url, server) = model_server(vec![action(
        "resource.update_resource",
        &[("ref", "release-notes"), ("body", "# Updated")],
    )])
    .await;
    let config = test_config(&root, &url);
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let response = agent.chat(request("update release notes", 1)).await;
    assert_eq!(response.stop_reason, "confirmation_required");
    assert!(!response
        .events
        .iter()
        .any(|event| event.kind == "tool_call"));
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn confirmed_resource_mutation_executes_stored_pending_tool() {
    let root = temp_root();
    let (model_url, model) = model_server(vec![confirm_action()]).await;
    let (resource_url, seen, resource) = resource_server().await;
    let mut config = test_config(&root, &model_url);
    config.kjxlkj_api_url = resource_url;
    let agent = Agent::new(config.clone(), ModelClient::from_config(&config));
    let first = agent.chat(request("update release notes", 2)).await;
    assert_eq!(first.stop_reason, "confirmation_required");
    let second = agent.chat(request("yes", 2)).await;
    assert_eq!(second.stop_reason, "finish");
    assert!(second.events.iter().any(|event| {
        event.kind == "tool_call" && event.tool.as_deref() == Some("resource.update_resource")
    }));
    let requests = seen.lock().await.join("\n");
    assert!(requests.contains("PUT /api/resources/release-notes "));
    model.abort();
    resource.abort();
    std::fs::remove_dir_all(root).unwrap();
}
