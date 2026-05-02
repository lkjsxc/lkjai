use clap::Parser;
use std::fs;
use std::path::PathBuf;

use crate::digest::{digest_bytes, digest_packed_data, digest_path, digest_source, write_file};
use crate::error::{Error, Result};
use crate::metadata::{load_tokenizer, read_config, Metadata};
use crate::source::encode_source_tokens;
use crate::FORMAT;

#[derive(Parser, Clone)]
pub(crate) struct BuildArgs {
    #[arg(long)]
    pub(crate) source: PathBuf,
    #[arg(long)]
    pub(crate) tokenizer: PathBuf,
    #[arg(long)]
    pub(crate) config: PathBuf,
    #[arg(long)]
    pub(crate) out: PathBuf,
    #[arg(long)]
    pub(crate) split: String,
    #[arg(long)]
    pub(crate) objective: String,
    #[arg(long)]
    pub(crate) seq_len: usize,
    #[arg(long)]
    pub(crate) sequence_count: usize,
    #[arg(long)]
    pub(crate) seed: u64,
    #[arg(long)]
    pub(crate) run_id: String,
}

pub(crate) fn build_cache(args: &BuildArgs) -> Result<Metadata> {
    if args.objective != "causal_lm_full" {
        return Err(Error::Message(format!(
            "unsupported objective: {}",
            args.objective
        )));
    }
    if args.seq_len <= 1 || args.sequence_count == 0 {
        return Err(Error::Message(
            "seq-len must be greater than 1 and sequence-count must be positive".into(),
        ));
    }
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let config = read_config(&args.config)?;
    let tokenizer_vocab = tokenizer.get_vocab_size(true) as u64;
    if tokenizer_vocab != config.vocab_size {
        return Err(Error::Message(format!(
            "tokenizer/config vocab mismatch: tokenizer={} config={}",
            tokenizer_vocab, config.vocab_size
        )));
    }
    if args.seq_len as u64 != config.context {
        return Err(Error::Message(format!(
            "seq_len/config context mismatch: seq_len={} context={}",
            args.seq_len, config.context
        )));
    }
    let source_digest = digest_source(&args.source)?;
    let tokenizer_digest = digest_path(&args.tokenizer)?;
    let config_digest = digest_path(&args.config)?;
    let needed = args
        .seq_len
        .checked_mul(args.sequence_count)
        .ok_or_else(|| Error::Message("seq-len * sequence-count overflows usize".into()))?;
    let (ids, example_count) =
        encode_source_tokens(&args.source, &tokenizer, &args.tokenizer, needed)?;
    if ids.len() < needed {
        return Err(Error::Message(format!(
            "source produced {} tokens, need {} for fixed non-overlapping windows",
            ids.len(),
            needed
        )));
    }
    let max_token_id = ids.iter().copied().max().unwrap_or(0) as u64;
    if max_token_id >= tokenizer_vocab || max_token_id >= config.vocab_size {
        return Err(Error::Message(format!(
            "token id {} outside tokenizer/config vocab {}",
            max_token_id, config.vocab_size
        )));
    }
    if max_token_id > u16::MAX as u64 {
        return Err(Error::Message(format!(
            "token id {} cannot be stored as uint16",
            max_token_id
        )));
    }
    fs::create_dir_all(&args.out).map_err(|source| Error::Io {
        path: args.out.clone(),
        source,
    })?;
    let mut tokens_bytes = Vec::with_capacity(needed * 2);
    for id in &ids {
        tokens_bytes.extend_from_slice(&(*id as u16).to_le_bytes());
    }
    let loss_mask_bytes = vec![1u8; needed];
    let starts_bytes = starts_bytes(args.sequence_count, args.seq_len);
    write_file(&args.out.join("tokens.bin"), &tokens_bytes)?;
    write_file(&args.out.join("loss_mask.bin"), &loss_mask_bytes)?;
    write_file(&args.out.join("starts.bin"), &starts_bytes)?;
    let metadata = metadata(
        args,
        config.vocab_size,
        &config_digest,
        &tokenizer_digest,
        &source_digest,
        example_count,
        needed,
        max_token_id,
        &tokens_bytes,
        &loss_mask_bytes,
        &starts_bytes,
    );
    write_file(
        &args.out.join("metadata.json"),
        format!("{}\n", serde_json::to_string_pretty(&metadata).unwrap()).as_bytes(),
    )?;
    Ok(metadata)
}

fn starts_bytes(sequence_count: usize, seq_len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(sequence_count * 8);
    for index in 0..sequence_count {
        bytes.extend_from_slice(&((index * seq_len) as u64).to_le_bytes());
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
fn metadata(
    args: &BuildArgs,
    vocab_size: u64,
    config_digest: &str,
    tokenizer_digest: &str,
    source_digest: &str,
    example_count: u64,
    needed: usize,
    max_token_id: u64,
    tokens: &[u8],
    loss_mask: &[u8],
    starts: &[u8],
) -> Metadata {
    Metadata {
        format: FORMAT.into(),
        schema_version: 2,
        split: args.split.clone(),
        objective: args.objective.clone(),
        sequence_len: args.seq_len as u64,
        seq_len: args.seq_len as u64,
        vocab_size,
        token_dtype: "uint16".into(),
        row_count: args.sequence_count as u64,
        sequence_count: args.sequence_count as u64,
        example_count,
        token_count: needed as u64,
        tokenizer_digest: tokenizer_digest.into(),
        config_digest: config_digest.into(),
        source_digest: source_digest.into(),
        seed: args.seed,
        run_id: args.run_id.clone(),
        max_token_id,
        tokens_checksum: digest_bytes(tokens),
        loss_mask_checksum: digest_bytes(loss_mask),
        starts_checksum: digest_bytes(starts),
        packed_data_checksum: digest_packed_data(tokens, loss_mask, starts),
    }
}
