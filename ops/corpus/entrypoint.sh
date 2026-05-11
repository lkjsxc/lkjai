#!/bin/sh
set -eu

cmd="${1:-help}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$cmd" in
  download-public-pretrain)
    exec /usr/local/bin/lkjai-public-pretrain download "$@"
    ;;
  prepare-public-pretrain)
    exec /usr/local/bin/lkjai-public-pretrain prepare "$@"
    ;;
  validate-public-pretrain)
    exec /usr/local/bin/lkjai-public-pretrain validate "$@"
    ;;
  build-tokenizer)
    tokenizer="${TRAIN_TOKENIZER_JSON:-/app/data/train/tokenizer/tokenizer.json}"
    max_vocab_size="${TRAIN_TOKENIZER_MAX_VOCAB_SIZE:-8192}"
    exec lkjai-native-tokenizer-build \
      --out "$tokenizer" \
      --max-vocab-size "$max_vocab_size"
    ;;
  build-public-pretrain-cache)
    source_jsonl="${TRAIN_PACKED_CACHE_SOURCE:-}"
    if [ -z "$source_jsonl" ]; then
      source_jsonl="${TRAIN_CORPUS_DIR:-/app/data/public-corpus}/train"
    fi
    tokenizer="${TRAIN_TOKENIZER_JSON:-/app/data/train/tokenizer/tokenizer.json}"
    config="${TRAIN_NATIVE_CONFIG:-/workspace/configs/native/decoder_40m_bf16_3070.json}"
    out="${TRAIN_PACKED_CACHE_DIR:-/app/data/train/datasets/packed/train-causal_lm_full-seq1024}"
    seq_len="${TRAIN_PACKED_CACHE_SEQ_LEN:-1024}"
    if [ ! -e "$source_jsonl" ]; then
      echo "missing TRAIN_PACKED_CACHE_SOURCE: $source_jsonl" >&2
      exit 2
    fi
    if [ ! -f "$tokenizer" ]; then
      echo "missing TRAIN_TOKENIZER_JSON: $tokenizer" >&2
      exit 2
    fi
    exec lkjai-native-packed-cache build \
      --source "$source_jsonl" \
      --tokenizer "$tokenizer" \
      --config "$config" \
      --out "$out" \
      --split train \
      --objective causal_lm_full \
      --seq-len "$seq_len" \
      --run-id public-pretrain
    ;;
  lkjai-native-packed-cache)
    exec lkjai-native-packed-cache "$@"
    ;;
  lkjai-native-tokenizer-build)
    exec lkjai-native-tokenizer-build "$@"
    ;;
  help|--help|-h)
    cat <<'USAGE'
usage: corpus COMMAND

commands:
  download-public-pretrain       fetch pinned public-pretrain Parquet files
  prepare-public-pretrain        write text-only JSONL train/val/holdout shards
  validate-public-pretrain       validate manifests and generated JSONL rows
  build-tokenizer                write native byte-level BPE tokenizer.json
  build-public-pretrain-cache    run native lkjai-packed-cache on JSONL source
USAGE
    ;;
  *)
    echo "unknown corpus command: $cmd" >&2
    exit 2
    ;;
esac
