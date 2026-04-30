# Quality

## Required Gates

- JSONL parses with one object per line.
- Required fields exist for the row mode.
- Active row language is `en`.
- Pretraining rows are standalone documents, not chat transcripts.
- SFT assistant messages are valid XML actions.
- SFT rows end with `agent.finish`.
- Preference-pair rows are excluded from active SFT.
- Metadata declares `kimi_synthetic` or `kimi-generated` provenance.
- Metadata declares `template_family`, `scenario_family_id`, `intent`,
  `tool_sequence`, `confirmation_required`, `grounding_source`, and
  `gold_stop_reason`.
- Resource mutation rows contain `agent.request_confirmation` before
  `resource.create_note`, `resource.create_media`, or `resource.update_resource`.
- Scenario families and near-duplicate clusters stay inside one split.
- Duplicate and near-duplicate rates stay low.
- Repetition, boilerplate, wrapper leakage, unsafe operational text, and API log
  contamination are flagged.

## Token Accounting

Prefer the repository tokenizer when `data/train/tokenizer/tokenizer.json`
exists. Otherwise use the documented approximation:

```text
approx_tokens = max(1, len(text) // 4)
```

Reports separate:

- pretraining approximate tokens,
- SFT total approximate tokens,
- SFT supervised approximate tokens,
- tokenizer-counted tokens when available.

## Sample Acceptance

A sample is acceptable when:

- it includes at least 20 SFT rows,
- only English rows are present,
- at least five domains are present,
- duplicate and near-duplicate rates are near zero,
- malformed JSONL count is zero.

## Pilot Acceptance

A committed pilot is acceptable when:

- XML validity is `1.0`,
- final `agent.finish` rate is `1.0`,
- duplicate rate is at most `0.01`,
- all four template families are present,
- every family has holdout rows,
- confirmation coverage for resource mutations is `1.0`.

## Full-Run Acceptance

A committed shard batch is acceptable when:

- all committed shards are `valid`,
- quarantined shards are excluded from training paths,
- manifest token totals and validation report agree,
- generated data is committed only after scoring,
- the background run can resume without overwriting valid shards,
- the pilot acceptance gates already passed,
- total tokenizer tokens reach `60000000`,
- duplicate rate is at most `0.01`.
