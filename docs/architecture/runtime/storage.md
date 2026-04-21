# Runtime Storage

## Root

- `DATA_DIR` defaults to `/app/data` in containers.
- Local Compose mounts project `./data` to `/app/data`.

## Directories

- `corpus/`: raw and tokenized corpus artifacts.
- `tokenizers/`: tokenizer artifacts.
- `checkpoints/`: training checkpoints.
- `models/`: serving exports.
- `runs/`: training and verification run logs.
- `agent/`: chat and tool transcripts.

## Persistence

- `data/` is untracked except for placeholders.
- Runtime code creates missing directories on boot.
