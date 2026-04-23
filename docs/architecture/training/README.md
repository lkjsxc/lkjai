# Training Architecture

Use this subtree for tokenizer training, scratch model training, agent
trajectory supervision, CUDA behavior, and export behavior.

## Read This Section When

- You need dataset and tokenizer formats.
- You need scratch-training pipeline commands.
- You need RTX 3070 CUDA expectations.

## Child Index

- [corpus.md](corpus.md): instruction and trajectory dataset contract
- [source-corpus.md](source-corpus.md): editable JSON source-entry contract
- [pipeline.md](pipeline.md): tokenizer, scratch training, eval, and export pipeline
- [cuda.md](cuda.md): GPU-first training behavior
- [dataset.md](dataset.md): dataset schema, generation, and validation
- [evaluation.md](evaluation.md): fixed eval and competency gate contract
- [preference.md](preference.md): preference-pair and DPO post-training rules

Operational runbooks for six-hour defaults and competency gates are owned by
[operations/training/README.md](../../operations/training/README.md).
