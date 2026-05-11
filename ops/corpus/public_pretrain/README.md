# Public Pretrain Tool Modules

## Purpose

Small Python modules for the corpus-only public pretraining helper.

## Contents

- [auth.py](auth.py): Hugging Face token resolution.
- [cli.py](cli.py): command dispatch.
- [common.py](common.py): shared paths, JSON, hashing, and token estimates.
- [download.py](download.py): pinned Parquet discovery and download.
- [prepare.py](prepare.py): text-only JSONL shard materialization.
- [shards.py](shards.py): JSONL shard writer.
- [validate.py](validate.py): manifest and generated-row validation.

## Boundary

This package is copied only into the `corpus` image. Product training, serving,
runtime, verification, and benchmark paths must remain native C++/CUDA.
