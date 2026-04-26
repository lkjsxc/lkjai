use lkjai::config::Config;

#[test]
fn config_has_local_bind_default() {
    std::env::remove_var("APP_HOST");
    std::env::remove_var("APP_PORT");
    std::env::remove_var("DATA_DIR");
    std::env::remove_var("MODEL_API_URL");
    std::env::remove_var("MODEL_NAME");
    std::env::remove_var("TOOL_WORKSPACE_DIR");
    std::env::remove_var("KJXLKJ_API_URL");
    std::env::remove_var("KJXLKJ_SESSION_COOKIE");
    let config = Config::from_env();
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8080);
    assert_eq!(
        config.model_api_url,
        "http://127.0.0.1:8081/v1/chat/completions"
    );
    assert_eq!(config.model_name, "lkjai-scratch-40m");
    assert_eq!(config.agent_max_steps, 6);
    assert_eq!(config.kjxlkj_api_url, "http://127.0.0.1:8080");
    assert_eq!(config.kjxlkj_session_cookie, "");
    assert_eq!(
        config.tool_workspace_dir,
        std::path::PathBuf::from("/app/data/workspace")
    );
}
