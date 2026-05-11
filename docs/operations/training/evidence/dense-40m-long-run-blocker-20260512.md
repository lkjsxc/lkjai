# Dense 40M Long-Run Blocker

Owner: `docs/operations/training/evidence/dense-40m-long-run-blocker-20260512.md`.
State: evidence note.

## Summary

The accepted dense 40M long run was not started. The local packed-cache path
required by `configs/training/dense_40m_accepted_3070.json` currently contains
a smoke fixture, not a valid seq1024/vocab8192 training cache.

## Observed Cache

Path:
`data/train/datasets/packed/train-causal_lm_full-seq1024/metadata.json`

Observed metadata:

```json
{
  "format": "lkjai-packed-cache",
  "split": "train",
  "objective": "causal_lm_full",
  "sequence_len": 16,
  "vocab_size": 256,
  "smoke_fixture": true,
  "token_dtype": "uint16",
  "row_count": 2,
  "token_count": 32
}
```

## Required Before Run

- Rebuild or validate a packed cache with `sequence_len=1024`.
- Use the dense 40M tokenizer/vocab contract with `vocab_size=8192`.
- Re-run
  `docker compose --progress quiet --profile verify run --build --rm verify`.
- Start the long run only after the cache and tokenizer checks pass.

No generated model artifacts, optimizer state, or fabricated metrics were
committed for this blocked run.
