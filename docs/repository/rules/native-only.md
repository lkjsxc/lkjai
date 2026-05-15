# Native Only

Owner: `docs/repository/rules/native-only.md`.
State: canonical documentation.

## Product Workflow

- Product training, serving, runtime, verification, and benchmark tooling must
  not depend on Rust or Python.
- Native C++/CUDA owns product binaries, model artifacts, tokenizer execution,
  inference routes, and repository checks.
- Python and Hugging Face dependencies are allowed only in isolated corpus
  acquisition paths.

## Node Rule

- Do not add `package.json`.
- Do not add Node-based verification.
- Do not install Node in Dockerfiles.
