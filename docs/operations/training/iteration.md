# Training Iteration Log

## Goal

Keep model-improvement claims tied to real artifacts, commands, and raw
behavioral reports.

## Baseline

- Artifact root: `data/models/lkjai-scratch-40m/`.
- Training summary: `data/train/checkpoints/training-summary.json`.
- Active default parameter target: about `40M`.
- Current behavioral report: `data/train/runs/behavioral-eval.json`.
- Current pass rate: `0.0` from `0/200` cases.
- Current issue: malformed or prompt-copy generations were wrapped into valid
  fallback final actions, inflating XML validity.
- Current artifacts were trained on disallowed LLM-authored corpus content and
  are invalid for acceptance after the provenance policy change.
- New baseline target: `500000000` public English pretraining tokens plus
  `60000000` first-party XML-action SFT tokens.

Materialize the ignored corpus after downloading Cosmopedia to
`data/raw/cosmopedia/`:

```bash
docker compose --profile corpus run --rm corpus download-public-pretrain
docker compose --profile corpus run --rm corpus prepare-public-pretrain
```

## Iteration Command

Fresh 500M-target run, started from an empty data directory:

```bash
docker compose --profile train run --rm \
  -e DATA_DIR=/app/data/train-full-500m-from-scratch-v2 \
  -e TRAIN_RESUME=never \
  -e TRAIN_INIT_CHECKPOINT= \
  -e TRAIN_CORPUS_DIR=/app/data/public-corpus \
  -e TRAIN_PUBLIC_DATA_DIR=/app/data/raw/cosmopedia \
  -e TRAIN_PUBLIC_PRETRAIN_TOKENS=500000000 \
  -e TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json \
  -e TRAIN_PRESET=agent \
  train train
```

Stopped first attempt, because it used the previous `440000000` public target:

- Step `1`: loss `9.1082`, `8192` input tokens seen.
- Step `3000`: loss `6.5823`, `24576000` input tokens seen.
- Step `6000`: loss `6.9035`, `49152000` input tokens seen.

Corrected 500M public run:

- Data directory: `data/train-full-500m-from-scratch-v2/`.
- Public train tokenizer tokens: `463087933`.
- Step `1`: loss `9.10818`, `8192` input tokens seen.
- Step `3000`: loss `6.9528`, `24576000` input tokens seen.
- Step `6000`: loss `7.1408`, `49152000` input tokens seen.
- Step `9000`: loss `7.3577`, `73728000` input tokens seen.
- Step `12000`: loss `7.4170`, `98304000` input tokens seen.
- Step `30000`: loss `7.2934`, `245760000` input tokens seen.
- Step `54000`: loss `7.3923`, `442368000` input tokens seen.
- Step `60000`: loss `7.4206`, `491520000` input tokens seen.

Two-hour finish adjustment:

- Stopped open-ended `400000`-step run after latest checkpoint step `75000`.
- Resumed causal-LM pretrain to bounded final step `84000`.
- Ran SFT from pretrain final checkpoint for bounded `12000` optimizer steps.
- SFT final loss: `3.5202`; validation loss: `3.5870`.
- Export: `data/train-full-500m-from-scratch-v2/exports/manifest.json`.
- Fixed eval: `16/16`.
- Generation sanity: `2/2` valid XML through structural-token suppression and
  invalid-generation fallback.
- Behavioral eval: `0/200`; XML validity `200/200`.

Overnight continuation:

- Container: `lkjai-sft-overnight-270k`.
- Resume source: latest SFT checkpoint from `data/train-full-500m-from-scratch-v2`.
- Target: continue `assistant_masked_sft` to optimizer step `270000`.
- Expected duration from step `30000`: about `11` hours at roughly `6`
  optimizer steps per second.
- Post-process watcher: `/tmp/lkjai-overnight-postprocess.log`.
- On successful training exit, the watcher exports best checkpoint and runs
  generation sanity, fixed eval, and behavioral eval.

Overnight result:

- Completed at optimizer step `270000`.
- Final SFT loss: `3.7791`.
- Best validation loss: `2.7690`.
- Final validation loss: `3.8169`.
- Fresh export completed at `2026-04-29 11:41`.
- Generation sanity: `2/2` valid XML.
- Fixed eval: `16/16`.
- Behavioral eval: `0/200`; XML validity `200/200`.
- Current generated content still uses the fallback `agent.finish` response:
  `I could not produce a complete model action.`
- Added protocol-constrained runtime decoding that fixes only the XML envelope
  and lets the local model generate `<content>`. This helps web usability while
  raw generation probes remain the acceptance signal.

## XML Repair Follow-up

High-LR repair run:

- Data root: `data/train-xml-repair-v1`.
- Init checkpoint: `data/train-full-500m-from-scratch-v2/checkpoints/best`.
- Stopped at optimizer step `60000` after a clean latest save.
- Best validation loss was at step `24000`: `2.0817`.
- Step `60000` raw probes still failed to emit complete `<action>` XML.

Low-LR repair run:

- Data root: `data/train-xml-repair-lr-v2`.
- Init checkpoint: `data/train-xml-repair-v1/checkpoints/best`.
- Learning rate: `0.00003` with `TRAIN_LR_MIN_FACTOR=0.2`.
- At step `30000`, validation loss reached about `1.9337`; best observed was
  about `1.9198` at step `21000`.
- Raw probes for `Hello`, `What is 2 + 2?`, and `Thanks` still failed complete
  XML; protocol decoding produced valid XML envelopes but poor content.
- Completed at step `60000`.
- Final validation loss: `1.9228`; best validation loss: `1.9167`.
- Final evals on the best export:
  - Generation sanity: `2/2` XML-valid through protocol decoding.
  - Fixed eval: `9/16`.
  - Behavioral eval: `0/200`; XML validity `200/200`.
- Do not accept either repair run until raw XML and content quality pass.

## Acceptance Record

Each accepted run records:

- command and environment overrides,
- checkpoint source,
- fixed eval pass rate,
- behavioral pass rate,
- XML validity without fallback wrapping,
- direct-answer, tool-call, confirmation, safety, and agentic bucket rates,
- manual inference probes.

## Speed Implementation Check

Performance implementation batch on 2026-04-30:

- Compose verify passed after the performance refactor.
- Training image builds on PyTorch `2.11.0+cu128` with CUDA `12.8`.
- FlashAttention is opt-in with `INSTALL_FLASH_ATTN=1`; the default image uses
  PyTorch SDPA paths so builds stay reliable.
- Synthetic 40M one-step CUDA check passed with batch `1`, compile off, and
  peak CUDA allocation about `675 MB`.
- Bounded synthetic GPU benchmark `speed-smoke/synthetic_gpu` passed.
- Auto-batch selected batch `8` for the bounded synthetic benchmark.
- Median profiled throughput in that benchmark was about `69910` input
  tokens/sec.
- This is a synthetic model-side smoke benchmark, not a replacement for the
  required real-data full training run.

Full retrain target:

```bash
docker compose --profile train run --rm \
  -e TRAIN_DATA_DIR=/app/data/train-speed-v1 \
  -e TRAIN_RESUME=never \
  -e TRAIN_INIT_CHECKPOINT= \
  train train
```

## Manual Probe Set

- `Say hello.`
- `List files in the workspace.`
- `Remember that I prefer concise plans.`
- `Search kjxlkj resources for release notes.`
- `Create a kjxlkj note with body "# Draft".`
- `Plan how to fix a failing test, then run the fix.`
- `Search docs for the deployment contract, then summarize it.`

## Update Rule

- Raise `TRAIN_BEHAVIORAL_THRESHOLD` after an accepted improvement.
- Do not lower the threshold to accept a weaker run.
- Do not record a run as improved unless raw generated actions beat the previous
  best pass rate.
