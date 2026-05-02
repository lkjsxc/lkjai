use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::digest::read_to_string;
use crate::error::{Error, Result};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Metadata {
    pub(crate) format: String,
    pub(crate) schema_version: u32,
    pub(crate) split: String,
    pub(crate) objective: String,
    pub(crate) sequence_len: u64,
    pub(crate) seq_len: u64,
    pub(crate) vocab_size: u64,
    pub(crate) token_dtype: String,
    pub(crate) row_count: u64,
    pub(crate) sequence_count: u64,
    pub(crate) example_count: u64,
    pub(crate) token_count: u64,
    pub(crate) tokenizer_digest: String,
    pub(crate) config_digest: String,
    pub(crate) source_digest: String,
    pub(crate) seed: u64,
    pub(crate) run_id: String,
    pub(crate) max_token_id: u64,
    pub(crate) tokens_checksum: String,
    pub(crate) loss_mask_checksum: String,
    pub(crate) starts_checksum: String,
    pub(crate) packed_data_checksum: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NativeConfig {
    pub(crate) vocab_size: u64,
    pub(crate) context: u64,
}

pub(crate) fn read_config(path: &Path) -> Result<NativeConfig> {
    let text = read_to_string(path)?;
    serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

pub(crate) fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|message| Error::Tokenizer {
        path: path.to_path_buf(),
        message: message.to_string(),
    })
}
