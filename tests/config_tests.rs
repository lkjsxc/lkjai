use lkjai::config::Config;

#[test]
fn config_has_local_bind_default() {
    std::env::remove_var("APP_HOST");
    std::env::remove_var("APP_PORT");
    std::env::remove_var("DATA_DIR");
    std::env::remove_var("MODEL_DIR");
    std::env::remove_var("INFERENCE_DEVICE");
    let config = Config::from_env();
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8080);
    assert_eq!(
        config.model_dir,
        std::path::PathBuf::from("/app/data/train/models/lkj-150m")
    );
    assert_eq!(config.inference_device, "cuda");
}
