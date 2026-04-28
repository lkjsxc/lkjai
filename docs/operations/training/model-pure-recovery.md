# Model-Pure Chat Recovery

## Goal

Restore smooth chat without adding canned replies, exact prompt lookup, XML
wrapping, or an external fallback model.

## Current Finding

As of 2026-04-28:

- Compose verify passes.
- Runtime transcripts show step-1 parse failures such as `missing <action>`.
- The exported checkpoint produces invalid XML for simple greetings.
- `data/train/runs/behavioral-eval.json` reports `pass_rate=0.0` and
  `xml_validity=0.0`.
- The active trained dataset has no `everyday_chat` rows.
- The exported checkpoint was trained with `causal_lm_full`, not the required
  `assistant_masked_sft` follow-up stage.

## Recovery Rules

- Do not add deterministic conversational fallback behavior.
- Do not wrap malformed generation into a valid `agent.finish` action.
- Do not use a stronger hosted model as the accepted runtime path.
- Fix prompt serialization, corpus mix, staged training, and acceptance gates.
- Export only a checkpoint that passes raw generation gates.

## Required Pipeline

1. Validate docs, sources, committed Kimi corpus, and public pretrain sources.
2. Prepare the public pretraining corpus if raw public data is present.
3. Train or resume the causal-LM pretrain stage.
4. Start `assistant_masked_sft` from the accepted pretrain weights.
5. Train SFT on first-party XML action traces with everyday chat included.
6. Export the accepted SFT checkpoint.
7. Run generation sanity and behavioral eval on the exported model.
8. Accept only if XML validity and chat pass-rate thresholds are met.

## Acceptance Probes

Manual browser/API probes must pass without transcript `error` events:

- `Hello`
- `What can you help me with?`
- `Thanks`
- `Remember that I prefer short answers.`
- `What do I prefer?`
- `List files in the workspace.`

The expected successful stop reason is `finish`, except tool tasks may use
intermediate tool events before the final `agent.finish`.
