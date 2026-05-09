#!/bin/sh
set -eu

check_bin="$1"
root="${TMPDIR:-/tmp}/lkjai-config-contract-negatives"
log="$root/check.log"

rm -rf "$root"
mkdir -p "$root/configs/native" "$root/configs/training"

cat > "$root/configs/native/decoder_40m_bf16_3070.json" <<'JSON'
{"model":"decoder-40m-bad","model_kind":"decoder","dtype":"bf16","vocab_size":512,"context":8,"layers":1,"hidden_size":28,"heads":4,"kv_heads":4,"head_dim":7,"ffn_size":64,"activation":"swiglu","rope_theta":10000,"rms_norm_eps":0.00001,"tie_embeddings":false,"seed":1337}
JSON
cat > "$root/configs/native/native_debug_bf16.json" <<'JSON'
{"model":"dense-bad","model_kind":"transformer","dtype":"bf16","vocab_size":512,"context":8,"layers":1,"hidden_size":28,"heads":4,"kv_heads":4,"head_dim":7,"ffn_size":64,"activation":"swiglu","rope_theta":10000,"rms_norm_eps":0.00001,"tie_embeddings":true,"seed":1337}
JSON
cat > "$root/configs/training/decoder_2h_40m_3070.json" <<'JSON'
{"format":"lkjai-train-config","name":"decoder_2h_40m_3070","description":"bad acceptance","preset":"decoder","model_name":"decoder","model_kind":"decoder","native_config":"configs/native/decoder_40m_bf16_3070.json","packed_cache_dir":"data/cache","tokenizer":"data/tokenizer.json","objective":"causal_lm_full","sequence_len":4,"learning_rate":0.001,"warmup_steps":0,"batch_size":1,"gradient_accumulation":1,"max_optimizer_steps":1,"save_latest_every_optimizer_steps":1,"target_seconds":7200,"seed":1}
JSON
cat > "$root/configs/training/profile_bad.json" <<'JSON'
{"format":"lkjai-train-config","name":"profile_bad","description":"bad profile","preset":"profile","model_name":"profile","model_kind":"decoder","native_config":"configs/native/decoder_40m_bf16_3070.json","packed_cache_dir":"data/cache","tokenizer":"data/tokenizer.json","objective":"causal_lm_full","sequence_len":4,"learning_rate":0.001,"warmup_steps":0,"batch_size":1,"gradient_accumulation":1,"max_optimizer_steps":1,"save_latest_every_optimizer_steps":1,"target_seconds":1,"seed":1}
JSON
cat > "$root/configs/training/bad_absolute.json" <<'JSON'
{"format":"lkjai-train-config","name":"bad_absolute","description":"bad native path","preset":"bad","model_name":"bad","model_kind":"dense","native_config":"/tmp/not-local.json","packed_cache_dir":"data/cache","tokenizer":"data/tokenizer.json","objective":"causal_lm_full","sequence_len":4,"learning_rate":0.001,"warmup_steps":0,"batch_size":1,"gradient_accumulation":1,"max_optimizer_steps":1,"save_latest_every_optimizer_steps":1,"target_seconds":1,"seed":1}
JSON

if "$check_bin" config-contract --repo "$root" > "$log" 2>&1; then
  echo "bad config contract unexpectedly passed" >&2
  exit 1
fi

grep -q "head_dim must be a multiple of 8" "$log"
grep -q "profile config uses acceptance decoder native config" "$log"
grep -q "native_config must be repo-local" "$log"
grep -q "missing \"tie_embeddings\": true" "$log"
