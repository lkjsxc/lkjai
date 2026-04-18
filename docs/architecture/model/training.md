# Training Contract

## Scope

- Train from random initialization.
- Build tokenizer and trainer in-repo.
- Keep CPU-only path first-class.

## Data Rules

- Corpora must be permissive and redistributable.
- Licenses must be documented in repository artifacts.
- Primary language is English with librarian/catalog content emphasis.

## Training Loop

1. Tokenize and shard corpus.
2. Build batch streams with deterministic seeds.
3. Execute forward/backward passes.
4. Apply optimizer update.
5. Emit checkpoints and evaluation metrics.

## Performance Rules

- Prefer contiguous memory layout and cache-aware kernels.
- Allow external CPU math libraries for matrix-heavy operations.
- Record throughput metrics per training stage.

