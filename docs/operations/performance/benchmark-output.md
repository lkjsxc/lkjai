# Benchmark Output

Native benchmark tools write JSONL for per-step samples and JSON/CSV summaries
for aggregate comparison.

## Step JSONL

Each step record includes:

- `step`
- `tokens`
- `optimizer_steps`
- `grad_accum`
- `microstep_seconds`
- `loader_seconds`
- `h2d_seconds`
- `forward_seconds`
- `backward_seconds`
- `optimizer_seconds`
- `loss`
- `capability`

`capability` uses the reusable native capability JSON shape.

## Summary JSON

Each summary includes:

- `commit`
- `config`
- `preset`
- `steps`
- `microsteps`
- `packed_cache_path`
- `median_tokens_per_second`
- `p95_microstep_seconds`
- `capability`
- `artifact_kind`
- `logits_checksum`

CSV summaries use the same stable names for columns that fit flat tabular
output. Nested capability fields are flattened with a `capability_` prefix.
