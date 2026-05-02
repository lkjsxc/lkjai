use clap::Parser;
use std::path::PathBuf;

use crate::digest::{
    digest_bytes, digest_packed_data, digest_path, digest_source, read_bytes, read_to_string,
};
use crate::error::{Error, Result};
use crate::metadata::{load_tokenizer, read_config, Metadata};
use crate::FORMAT;

#[derive(Parser, Clone)]
pub(crate) struct ValidateArgs {
    #[arg(long)]
    pub(crate) cache: PathBuf,
    #[arg(long)]
    pub(crate) config: PathBuf,
    #[arg(long)]
    pub(crate) tokenizer: PathBuf,
    #[arg(long)]
    pub(crate) source: PathBuf,
}

pub(crate) fn validate_cache(args: &ValidateArgs) -> Result<Metadata> {
    let metadata_path = args.cache.join("metadata.json");
    let text = read_to_string(&metadata_path)?;
    let metadata: Metadata = serde_json::from_str(&text).map_err(|source| Error::Json {
        path: metadata_path.clone(),
        source,
    })?;
    validate_metadata_shape(&metadata)?;
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let tokenizer_vocab = tokenizer.get_vocab_size(true) as u64;
    let config = read_config(&args.config)?;
    if metadata.tokenizer_digest != digest_path(&args.tokenizer)? {
        return Err(Error::Message("metadata tokenizer_digest is stale".into()));
    }
    if metadata.config_digest != digest_path(&args.config)? {
        return Err(Error::Message("metadata config_digest is stale".into()));
    }
    if metadata.source_digest != digest_source(&args.source)? {
        return Err(Error::Message("metadata source_digest is stale".into()));
    }
    if metadata.vocab_size != tokenizer_vocab || metadata.vocab_size != config.vocab_size {
        return Err(Error::Message(format!(
            "vocab mismatch: metadata={} tokenizer={} config={}",
            metadata.vocab_size, tokenizer_vocab, config.vocab_size
        )));
    }
    if metadata.seq_len != config.context {
        return Err(Error::Message(format!(
            "seq_len/config context mismatch: metadata={} context={}",
            metadata.seq_len, config.context
        )));
    }
    validate_payload(args, &metadata, tokenizer_vocab, config.vocab_size)?;
    Ok(metadata)
}

fn validate_metadata_shape(metadata: &Metadata) -> Result<()> {
    if metadata.format != FORMAT {
        return Err(Error::Message(
            "metadata format must be lkjai-packed-cache-v2".into(),
        ));
    }
    if metadata.token_dtype != "uint16" {
        return Err(Error::Message("metadata token_dtype must be uint16".into()));
    }
    if metadata.sequence_len == 0 || metadata.seq_len != metadata.sequence_len {
        return Err(Error::Message(
            "metadata seq_len must match sequence_len and be positive".into(),
        ));
    }
    if metadata.row_count != metadata.sequence_count {
        return Err(Error::Message(
            "metadata row_count must match sequence_count".into(),
        ));
    }
    if metadata.token_count != metadata.seq_len * metadata.sequence_count {
        return Err(Error::Message(
            "metadata token_count must equal seq_len * sequence_count".into(),
        ));
    }
    Ok(())
}

fn validate_payload(
    args: &ValidateArgs,
    meta: &Metadata,
    tokenizer_vocab: u64,
    config_vocab: u64,
) -> Result<()> {
    let tokens = read_bytes(&args.cache.join("tokens.bin"))?;
    let loss_mask = read_bytes(&args.cache.join("loss_mask.bin"))?;
    let starts = read_bytes(&args.cache.join("starts.bin"))?;
    if tokens.len() != meta.token_count as usize * 2 || tokens.len() % 2 != 0 {
        return Err(Error::Message(
            "tokens.bin size does not match metadata token_count".into(),
        ));
    }
    if loss_mask.len() != meta.token_count as usize {
        return Err(Error::Message(
            "loss_mask.bin size does not match token_count".into(),
        ));
    }
    if starts.len() != meta.sequence_count as usize * 8 {
        return Err(Error::Message(
            "starts.bin size does not match metadata sequence_count".into(),
        ));
    }
    if meta.tokens_checksum != digest_bytes(&tokens) {
        return Err(Error::Message("metadata tokens_checksum is stale".into()));
    }
    if meta.loss_mask_checksum != digest_bytes(&loss_mask) {
        return Err(Error::Message(
            "metadata loss_mask_checksum is stale".into(),
        ));
    }
    if meta.starts_checksum != digest_bytes(&starts) {
        return Err(Error::Message("metadata starts_checksum is stale".into()));
    }
    if meta.packed_data_checksum != digest_packed_data(&tokens, &loss_mask, &starts) {
        return Err(Error::Message(
            "metadata packed_data_checksum is stale".into(),
        ));
    }
    validate_tokens(&tokens, meta, tokenizer_vocab, config_vocab)?;
    validate_starts_and_mask(&starts, &loss_mask, meta)
}

fn validate_tokens(
    tokens: &[u8],
    meta: &Metadata,
    tokenizer_vocab: u64,
    config_vocab: u64,
) -> Result<()> {
    let mut max_token_id = 0u64;
    for chunk in tokens.chunks_exact(2) {
        let id = u16::from_le_bytes([chunk[0], chunk[1]]) as u64;
        max_token_id = max_token_id.max(id);
        if id >= meta.vocab_size || id >= tokenizer_vocab || id >= config_vocab {
            return Err(Error::Message(format!(
                "token id {} outside tokenizer/config vocab",
                id
            )));
        }
    }
    if max_token_id != meta.max_token_id {
        return Err(Error::Message("metadata max_token_id is stale".into()));
    }
    Ok(())
}

fn validate_starts_and_mask(starts: &[u8], loss_mask: &[u8], meta: &Metadata) -> Result<()> {
    for (index, chunk) in starts.chunks_exact(8).enumerate() {
        let offset = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)"));
        let expected = index as u64 * meta.seq_len;
        if offset != expected || offset + meta.seq_len > meta.token_count {
            return Err(Error::Message(
                "starts.bin contains invalid fixed window offsets".into(),
            ));
        }
    }
    if loss_mask.iter().any(|value| *value > 1) {
        return Err(Error::Message(
            "loss_mask.bin contains values other than 0 or 1".into(),
        ));
    }
    Ok(())
}
