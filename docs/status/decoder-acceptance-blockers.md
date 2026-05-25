# Decoder Acceptance Blockers

Owner: `docs/status/decoder-acceptance-blockers.md`.
State: acceptance blocker list.

## Current Non-Accepted Truth

`decoder_backward_backend=cuda_full_decoder` is valid for chain-rule CUDA
backward evidence. It does not make the run accepted while attention, decode,
shape, logits, deltas, or route evidence are missing.

## Required Promotion Evidence

Accepted status requires all of:

- cuDNN SDPA BF16 GQA forward and backward execution,
- zero reference-attention fallback in accepted mode,
- RTX 3070 40M shape and two-hour training config,
- accepted decode backend and KV backend,
- logits check pass,
- positive non-embedding and block weight deltas,
- accepted route transcript with matching report and artifact digests.

## Blocker Reporting

If GPU or route evidence is unavailable, report
`lkjai_decode_accepted=false` and include the exact command that failed or the
missing artifact path.
