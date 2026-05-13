# Keep And Remove

Owner: `docs/decisions/keep-remove.md`.
State: canonical scope decision list.

## Keep

- Docs-first canon and topology validation.
- Docker Compose profiles.
- JSONL transcript persistence.
- Local-only default bind.
- Host-YOLO tool concept with explicit transcript logging.
- Data-directory workspace boundary for file and shell tools.
- Docs link validation, line limits, and no-Node gate.
- Native C++/CUDA product binaries for train, serve, runtime, tooling, and
  verification.

## Remove Or Replace

- Qwen default serving and tuning assumptions.
- QLoRA, LoRA, PEFT, bitsandbytes, and adapter defaults.
- Pretrained serving models as the default runtime.
- `lkj-150m` as the immediate default serving target.
- FineWeb-Edu 3B-token default training run.
- 512 MiB artifact limit as a core success criterion.
- Direct natural-language tool routing as the main agent mechanism.
- Host root mounts for agent file and shell tools.
- Rust runtime/tool crates, Python tooling/tests, PyTorch checkpoints, and
  cookie/session resource integration.

## Reframe

- Model work becomes native tokenizer plus from-scratch dense LM training.
- Training quality becomes agent eval pass rate, not raw loss alone.
- Context handling becomes memory retrieval and summaries.
