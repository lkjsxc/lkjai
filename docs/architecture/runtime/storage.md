# Runtime Storage

## Root

- `DATA_DIR` defaults to `/app/data` in containers.
- Local Compose mounts project `./data` to `/app/data`.

## Directories

- `models/`: scratch serving artifacts and model manifests.
- `train/datasets/`: instruction and trajectory training datasets.
- `train/tokenizer/`: locally trained tokenizer artifacts.
- `train/checkpoints/`: scratch model checkpoints.
- `train/exports/`: serving manifests copied from training outputs.
- `train/runs/`: training and eval logs.
- `agent/runs/`: chat and tool transcripts.
- `agent/memory.sqlite3`: durable memory and summaries.

## Persistence

- `data/` is untracked except for placeholders.
- Runtime code creates missing directories on boot.
