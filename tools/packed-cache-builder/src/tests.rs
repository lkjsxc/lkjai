use super::*;
use crate::digest::{read_bytes, read_to_string, write_file};
use tempfile::TempDir;

struct Fixture {
    _tmp: TempDir,
    source: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
    config: std::path::PathBuf,
    cache: std::path::PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    let source = root.join("train.jsonl");
    let tokenizer = root.join("tokenizer.json");
    let config = root.join("native.json");
    let cache = root.join("cache");
    write_file(
        &source,
        br#"{"id":1,"text":"hello world alpha beta gamma delta"}
{"messages":[{"content":"hello world"},{"content":"alpha beta gamma delta hello world"}]}
"#,
    )
    .expect("source");
    write_file(&tokenizer, tokenizer_json().as_bytes()).expect("tokenizer");
    write_file(
        &config,
        br#"{"model":"test","dtype":"bf16","vocab_size":8,"context":4}"#,
    )
    .expect("config");
    Fixture {
        _tmp: tmp,
        source,
        tokenizer,
        config,
        cache,
    }
}

fn tokenizer_json() -> String {
    r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2,
      "alpha": 3,
      "beta": 4,
      "gamma": 5,
      "delta": 6,
      "unused": 7
    },
    "unk_token": "[UNK]"
  }
}"#
    .into()
}

fn build_args(f: &Fixture) -> build::BuildArgs {
    build::BuildArgs {
        source: f.source.clone(),
        tokenizer: f.tokenizer.clone(),
        config: f.config.clone(),
        out: f.cache.clone(),
        split: "train".into(),
        objective: "causal_lm_full".into(),
        seq_len: 4,
        sequence_count: 2,
        seed: 20260502,
        run_id: "test-run".into(),
    }
}

fn validate_args(f: &Fixture) -> validate::ValidateArgs {
    validate::ValidateArgs {
        cache: f.cache.clone(),
        config: f.config.clone(),
        tokenizer: f.tokenizer.clone(),
        source: f.source.clone(),
    }
}

#[test]
fn deterministic_rebuild_produces_identical_metadata_and_checksums() {
    let f = fixture();
    let first = build::build_cache(&build_args(&f)).expect("first build");
    let metadata_first = read_to_string(&f.cache.join("metadata.json")).expect("metadata");
    let tokens_first = read_bytes(&f.cache.join("tokens.bin")).expect("tokens");
    let second = build::build_cache(&build_args(&f)).expect("second build");
    let metadata_second = read_to_string(&f.cache.join("metadata.json")).expect("metadata");
    let tokens_second = read_bytes(&f.cache.join("tokens.bin")).expect("tokens");
    assert_eq!(first.tokens_checksum, second.tokens_checksum);
    assert_eq!(metadata_first, metadata_second);
    assert_eq!(tokens_first, tokens_second);
    validate::validate_cache(&validate_args(&f)).expect("validate");
}

#[test]
fn rejects_vocab_mismatch() {
    let f = fixture();
    write_file(
        &f.config,
        br#"{"model":"test","dtype":"bf16","vocab_size":9,"context":4}"#,
    )
    .expect("config");
    let error = build::build_cache(&build_args(&f)).expect_err("vocab mismatch");
    assert!(error.to_string().contains("vocab mismatch"), "{error}");
}

#[test]
fn rejects_seq_len_mismatch() {
    let f = fixture();
    write_file(
        &f.config,
        br#"{"model":"test","dtype":"bf16","vocab_size":8,"context":8}"#,
    )
    .expect("config");
    let error = build::build_cache(&build_args(&f)).expect_err("seq mismatch");
    assert!(error.to_string().contains("seq_len/config"), "{error}");
}

#[test]
fn rejects_truncated_or_corrupt_binaries() {
    for name in ["tokens.bin", "loss_mask.bin", "starts.bin"] {
        let f = fixture();
        build::build_cache(&build_args(&f)).expect("build");
        let path = f.cache.join(name);
        let mut bytes = read_bytes(&path).expect("bytes");
        bytes.pop();
        write_file(&path, &bytes).expect("truncate");
        let error = validate::validate_cache(&validate_args(&f)).expect_err("truncated");
        assert!(
            error.to_string().contains(name) || error.to_string().contains("checksum"),
            "{name}: {error}"
        );
    }
}

#[test]
fn rejects_stale_metadata_checksum() {
    let f = fixture();
    build::build_cache(&build_args(&f)).expect("build");
    let mut tokens = read_bytes(&f.cache.join("tokens.bin")).expect("tokens");
    tokens[0] = 7;
    write_file(&f.cache.join("tokens.bin"), &tokens).expect("tokens");
    let error = validate::validate_cache(&validate_args(&f)).expect_err("stale checksum");
    assert!(error.to_string().contains("tokens_checksum"), "{error}");
}
