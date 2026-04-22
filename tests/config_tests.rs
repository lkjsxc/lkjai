use lkjai::config::Config;

#[test]
fn config_has_local_bind_default() {
    std::env::remove_var("APP_HOST");
    std::env::remove_var("APP_PORT");
    std::env::remove_var("DATA_DIR");
    std::env::remove_var("MODEL_API_URL");
    std::env::remove_var("MODEL_NAME");
    let config = Config::from_env();
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8080);
    assert_eq!(
        config.model_api_url,
        "http://127.0.0.1:8081/v1/chat/completions"
    );
    assert_eq!(config.model_name, "qwen3-1.7b-q4");
    assert_eq!(config.agent_max_steps, 6);
}
