# Runtime Storage

## Root

- `DATA_DIR` defaults to `/app/data` in containers.
- Local Compose mounts project `./data` to `/app/data`.

## Directories

- `train/corpus/`: raw and tokenized corpus artifacts.
- `train/tokenizers/`: tokenizer artifacts.
- `train/checkpoints/`: training checkpoints.
- `train/models/`: serving exports.
- `train/runs/`: training and fixed-eval logs.
- `agent/`: chat and tool transcripts.

## Persistence

- `data/` is untracked except for placeholders.
- Runtime code creates missing directories on boot.
