# Principles

## Build Principles

- Documentation defines behavior before implementation.
- Implementation follows the smallest coherent contract that satisfies the docs.
- GPU work is preferred over CPU work when it improves training or inference
  throughput without making verification brittle.
- Long-running training must be resumable.
- Verification stays small enough to run routinely.

## Product Principles

- The web app is local-first.
- Host-YOLO actions are explicit in transcripts.
- Tool execution favors operator power over sandboxing.
- Dangerous defaults must be documented plainly.

## Model Principles

- The v1 model is trained from scratch.
- The model architecture borrows proven small-model techniques without copying
  restricted weights.
- The training corpus is openly licensed and reproducible.
- The serving export is the artifact constrained to 512 MiB.
