use lkjai::config::Config;

#[test]
fn config_has_local_bind_default() {
    std::env::remove_var("APP_HOST");
    std::env::remove_var("APP_PORT");
    let config = Config::from_env();
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8080);
}
