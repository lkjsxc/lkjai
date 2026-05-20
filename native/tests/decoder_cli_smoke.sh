#!/bin/sh
set -eu

train_bin="$1"
tokenizer_bin="$2"
root="${TMPDIR:-/tmp}/lkjai-decoder-cli-smoke"
cache="$root/cache"
out="$root/out"
tokenizer="$root/tokenizer.json"
config="$root/decoder_debug_bf16.json"
report="$root/report.json"

rm -rf "$root"
mkdir -p "$cache"

cat > "$cache/metadata.json" <<'JSON'
{"format":"lkjai-packed-cache","split":"train","objective":"causal_lm_full","sequence_len":4,"vocab_size":512,"smoke_fixture":true,"token_dtype":"uint16","row_count":2,"token_count":8}
JSON
printf '\001\000\002\000\003\000\004\000\005\000\006\000\007\000\010\000' > "$cache/tokens.bin"
printf '\001\001\001\001\001\001\001\001' > "$cache/loss_mask.bin"
printf '\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000' > "$cache/starts.bin"

cat > "$config" <<'JSON'
{"model":"decoder-cli-smoke","model_kind":"decoder","dtype":"bf16","vocab_size":512,"context":8,"layers":1,"hidden_size":32,"heads":4,"kv_heads":4,"head_dim":8,"ffn_size":64,"activation":"swiglu","rope_theta":10000,"rms_norm_eps":0.00001,"tie_embeddings":true,"seed":1337}
JSON

"$tokenizer_bin" --out "$tokenizer" --max-vocab-size 512 >/dev/null

"$train_bin" --train --mode decoder --packed-cache "$cache" \
  --config "$config" --tokenizer "$tokenizer" --out "$out" \
  --max-steps 1 --seq-len 4 --batch-size 1 --grad-accum 1 \
  --run-purpose smoke > "$report"

grep -q '"status":"success"' "$report"
grep -q '"implementation_status":"experimental"' "$report"
grep -q '"accepted_cuda_training":false' "$report"
grep -q '"decoder_cuda_slice":"full_decoder"' "$report"
grep -q '"forward_backend":"cuda_full_decoder"' "$report"
grep -q '"backward_backend":"cuda_full_decoder"' "$report"
grep -q '"attention_backend":"cuda_causal_gqa_bf16_reference"' "$report"
grep -q '"decoder_weight_change"' "$report"
grep -q '"non_embedding_weight_changed":true' "$report"
grep -q '"decoder_block_weight_changed":true' "$report"
grep -q '"decoder_backward_backend":"not_accepted_cuda_full_decoder"' "$report"
grep -q '"decoder_gradient_source":"cuda_device"' "$report"
! grep -q '"accepted_cuda_training":true' "$report"
! grep -q '"attention_backend":"cudnn_sdpa_bf16_gqa"' "$report"
! grep -q '"decoder_gradient_source":"host_reference"' "$report"
! grep -q '"decoder_backward_diagnostic_synthetic"' "$report"
grep -q '"decode_backend":"cuda_reference_kv_cache"' "$report"
grep -q '"kv_cache_backend":"cuda_contiguous_bf16_partial"' "$report"
grep -q '"decode_supported":true' "$report"
test ! -f "$root/models/decoder-40m-3070/decoder_acceptance.json"
