import json
import os
from pathlib import Path

from dense_accepted_training_io import (
    CONFIG_HOST,
    SEED,
    SEQ_LEN,
    SOURCE_HOST,
    TOKENIZER_HOST,
    workspace_path,
)
from run_support import run


def cargo_builder(args: list[str], log_path: Path) -> None:
    command = [
        "docker",
        "compose",
        "--progress",
        "quiet",
        "--profile",
        "verify",
        "run",
        "--rm",
        "--entrypoint",
        "cargo",
        "verify",
        "run",
        "-p",
        "lkjai_packed_cache_builder",
        "--",
        *args,
    ]
    code = run(command, log_path, os.environ.copy())
    if code != 0:
        raise SystemExit(code)


def build_and_validate_cache(
    run_id: str, sequence_count: int, cache_dir: Path, out_dir: Path
) -> dict:
    base = [
        "--source",
        workspace_path(SOURCE_HOST),
        "--tokenizer",
        workspace_path(TOKENIZER_HOST),
        "--config",
        workspace_path(CONFIG_HOST),
    ]
    cargo_builder(
        [
            "build",
            *base,
            "--out",
            workspace_path(cache_dir),
            "--split",
            "train",
            "--objective",
            "causal_lm_full",
            "--seq-len",
            str(SEQ_LEN),
            "--sequence-count",
            str(sequence_count),
            "--seed",
            str(SEED),
            "--run-id",
            run_id,
        ],
        out_dir / "packed-cache-build.log",
    )
    cargo_builder(
        ["validate", "--cache", workspace_path(cache_dir), *base],
        out_dir / "packed-cache-validate.log",
    )
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    (out_dir / "packed-cache-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def cache_summary(metadata: dict) -> dict:
    return {
        "format": metadata.get("format", ""),
        "schema_version": int(metadata.get("schema_version", 0)),
        "split": metadata.get("split", ""),
        "objective": metadata.get("objective", ""),
        "sequence_len": int(metadata.get("sequence_len", metadata.get("seq_len", 0))),
        "row_count": int(metadata.get("row_count", 0)),
        "sequence_count": int(metadata.get("sequence_count", 0)),
        "token_count": int(metadata.get("token_count", 0)),
        "example_count": int(metadata.get("example_count", 0)),
        "seed": int(metadata.get("seed", 0)),
        "source_digest": metadata.get("source_digest", ""),
        "tokenizer_digest": metadata.get("tokenizer_digest", ""),
        "config_digest": metadata.get("config_digest", ""),
        "tokens_checksum": metadata.get("tokens_checksum", ""),
        "loss_mask_checksum": metadata.get("loss_mask_checksum", ""),
        "starts_checksum": metadata.get("starts_checksum", ""),
        "packed_data_checksum": metadata.get("packed_data_checksum", ""),
    }
