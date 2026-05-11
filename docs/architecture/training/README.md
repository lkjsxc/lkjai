# Training Architecture

Use this subtree for tokenizer training, scratch model training, agent
trajectory supervision, CUDA behavior, and export behavior.

## Read This Section When

- You need dataset and tokenizer formats.
- You need scratch-training pipeline commands.
- You need RTX 3070 CUDA expectations.

## Child Index

- [data/README.md](data/README.md): corpus, dataset, tokenizer, provenance,
  and packed-cache contracts.
- [pipeline/README.md](pipeline/README.md): scratch training pipeline and
  GPU-first training behavior.
- [evaluation/README.md](evaluation/README.md): fixed eval, competency gates,
  and preference-pair rules.

Operational runbooks for six-hour defaults and competency gates are owned by
[operations/training/README.md](../../operations/training/README.md).
