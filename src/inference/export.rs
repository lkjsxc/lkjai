use crate::config::Config;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct ExportManifest {
    format: String,
    config: String,
    weights: String,
    tokenizer: String,
    vocab_size: usize,
    context: usize,
    tokenizer_sha256: String,
}

#[derive(Deserialize)]
struct ConfigSummary {
    vocab_size: usize,
    context: usize,
}

pub fn tokenizer_path(config: &Config) -> Result<PathBuf, String> {
    let local = config.model_dir.join("tokenizer.json");
    if local.exists() {
        return Ok(local);
    }
    Err(format!("tokenizer missing: expected {}", display(&local)))
}

pub fn validate_export(config: &Config, tokenizer_path: &Path) -> Result<(), String> {
    let manifest_path = config.model_dir.join("manifest.json");
    let manifest: ExportManifest = read_json(&manifest_path)?;
    if manifest.format != "lkjai-export-v1" {
        return Err(format!(
            "unsupported export manifest format {}",
            manifest.format
        ));
    }
    for artifact in [&manifest.config, &manifest.weights, &manifest.tokenizer] {
        let path = config.model_dir.join(artifact);
        if !path.exists() {
            return Err(format!("export artifact missing: {}", display(&path)));
        }
    }
    if tokenizer_path.file_name().and_then(|name| name.to_str())
        != Some(manifest.tokenizer.as_str())
    {
        return Err("export manifest tokenizer does not match loaded tokenizer path".into());
    }
    let actual_hash = file_sha256(tokenizer_path)?;
    if actual_hash != manifest.tokenizer_sha256 {
        return Err("export tokenizer hash does not match manifest".into());
    }
    let cfg: ConfigSummary = read_json(&config.model_dir.join(&manifest.config))?;
    if cfg.vocab_size != manifest.vocab_size || cfg.context != manifest.context {
        return Err("export manifest does not match config.json".into());
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", display(path)))?;
    serde_json::from_str(&text).map_err(|e| format!("failed to parse {}: {e}", display(path)))
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|e| format!("failed to open {}: {e}", display(path)))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 64];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("failed to read {}: {e}", display(path)))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn display(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(root: &Path) -> Config {
        Config {
            host: "127.0.0.1".into(),
            port: 8080,
            data_dir: root.join("data"),
            model_dir: root.join("model"),
            inference_device: "cpu".into(),
            tool_timeout_secs: 20,
            tool_output_limit: 12_000,
        }
    }

    #[test]
    fn tokenizer_must_be_colocated_with_model() {
        let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
        let config = test_config(&root);
        std::fs::create_dir_all(config.data_dir.join("tokenizers")).unwrap();
        std::fs::write(config.data_dir.join("tokenizers/tokenizer.json"), "{}").unwrap();
        let error = tokenizer_path(&config).unwrap_err();
        assert!(error.contains("tokenizer missing: expected"));
        assert!(!error.contains("tokenizers/tokenizer.json or"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn export_validation_rejects_hash_mismatch() {
        let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
        let config = test_config(&root);
        std::fs::create_dir_all(&config.model_dir).unwrap();
        std::fs::write(
            config.model_dir.join("config.json"),
            r#"{"vocab_size":3,"context":4}"#,
        )
        .unwrap();
        std::fs::write(config.model_dir.join("model.safetensors"), "weights").unwrap();
        let tokenizer = config.model_dir.join("tokenizer.json");
        std::fs::write(&tokenizer, "tokenizer").unwrap();
        std::fs::write(config.model_dir.join("manifest.json"), mismatch_manifest()).unwrap();
        let error = validate_export(&config, &tokenizer).unwrap_err();
        assert!(error.contains("tokenizer hash"));
        std::fs::remove_dir_all(root).unwrap();
    }

    fn mismatch_manifest() -> &'static str {
        r#"{"format":"lkjai-export-v1","config":"config.json","weights":"model.safetensors","tokenizer":"tokenizer.json","vocab_size":3,"context":4,"tokenizer_sha256":"bad"}"#
    }
}
