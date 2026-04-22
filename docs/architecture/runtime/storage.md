# Runtime Storage

## Root

- `DATA_DIR` defaults to `/app/data` in containers.
- Local Compose mounts project `./data` to `/app/data`.

## Directories

- `models/`: GGUF models and model manifests.
- `train/datasets/`: instruction and trajectory tuning datasets.
- `train/adapters/`: LoRA and QLoRA adapters.
- `train/exports/`: merged and quantized tuning outputs.
- `train/runs/`: training and eval logs.
- `agent/runs/`: chat and tool transcripts.
- `agent/memory.sqlite3`: durable memory and summaries.

## Persistence

- `data/` is untracked except for placeholders.
- Runtime code creates missing directories on boot.
