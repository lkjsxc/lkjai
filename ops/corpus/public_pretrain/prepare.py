import argparse
import json
import os
from pathlib import Path

from .common import (
    DEFAULT_OUT,
    DEFAULT_RAW,
    DEFAULT_SOURCE,
    env_path,
    estimated_tokens,
    file_sha256,
    load_recipes,
    read_json,
    safe_command_meta,
    split_for_ordinal,
    write_json,
)
from .shards import ShardWriter, clean_output
from .validate import validate


def source_files(raw_dir, recipe, download_manifest):
    for source in download_manifest.get("sources", []):
        if source.get("config") != recipe["config"]:
            continue
        for item in source.get("files", []):
            path = Path(raw_dir) / recipe["config"] / "train" / item["filename"]
            if not path.is_file():
                raise RuntimeError(f"missing downloaded parquet: {path}")
            if file_sha256(path) != item["sha256"]:
                raise RuntimeError(f"checksum mismatch: {path}")
            yield path, item


def prepare(_args):
    source_file = env_path("CORPUS_SOURCE_FILE", DEFAULT_SOURCE)
    raw_dir = env_path("TRAIN_PUBLIC_DATA_DIR", DEFAULT_RAW)
    out_dir = env_path("TRAIN_CORPUS_DIR", DEFAULT_OUT)
    target = int(os.environ.get("TRAIN_PUBLIC_PRETRAIN_TOKENS", "500000000"))
    recipes = load_recipes(source_file)
    download_manifest = read_json(raw_dir / "download-manifest.json")
    clean_output(out_dir)
    writers = {s: ShardWriter(out_dir, s) for s in ("train", "val", "holdout")}
    manifest = new_manifest(target)
    try:
        fill_manifest(recipes, raw_dir, target, download_manifest, writers,
                      manifest)
    finally:
        for writer in writers.values():
            writer.close()
    write_json(out_dir / "manifest.json", manifest)
    validate(argparse.Namespace())
    print(json.dumps({"status": "pass",
                      "manifest": str(out_dir / "manifest.json")}))


def new_manifest(target):
    return {
        "schema": "lkjai-public-pretrain-corpus",
        "command": safe_command_meta("prepare-public-pretrain"),
        "field_policy": "text-only",
        "selected_fields": ["text"],
        "target_token_estimate": target,
        "token_estimate": 0,
        "row_count": 0,
        "split_rows": {"train": 0, "val": 0, "holdout": 0},
        "source_distribution": {},
        "sources": [],
    }


def fill_manifest(recipes, raw_dir, target, download_manifest, writers,
                  manifest):
    ordinal = 0
    for recipe in recipes:
        budget = min(int(recipe["token_budget"]),
                     target - manifest["token_estimate"])
        source_meta = source_meta_for(recipe, budget)
        ordinal = fill_source(raw_dir, recipe, download_manifest, writers,
                              manifest, source_meta, ordinal, budget)
        manifest["source_distribution"][recipe["name"]] = (
            source_meta["token_estimate"])
        manifest["sources"].append(source_meta)
        if manifest["token_estimate"] >= target:
            break


def source_meta_for(recipe, budget):
    return {
        "name": recipe["name"],
        "dataset": recipe["dataset"],
        "config": recipe["config"],
        "dataset_revision": recipe["revision"],
        "license": recipe["license"],
        "token_budget": budget,
        "token_estimate": 0,
        "row_count": 0,
        "files": [],
    }


def fill_source(raw_dir, recipe, download_manifest, writers, manifest,
                source_meta, ordinal, source_budget):
    columns = [recipe["text_field"]]
    if recipe.get("token_field"):
        columns.append(recipe["token_field"])
    for parquet_path, file_meta in source_files(raw_dir, recipe,
                                                download_manifest):
        file_rows, file_tokens, ordinal = fill_file(
            parquet_path, file_meta, recipe, writers, manifest, source_meta,
            ordinal, source_budget, columns)
        source_meta["files"].append({
            "filename": file_meta["filename"],
            "sha256": file_meta["sha256"],
            "row_count": file_rows,
            "token_estimate": file_tokens,
        })
        if reached_budget(manifest, source_meta, source_budget):
            break
    return ordinal


def fill_file(path, file_meta, recipe, writers, manifest, source_meta, ordinal,
              source_budget, columns):
    import pyarrow.parquet as pq

    file_rows = 0
    file_tokens = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        table = batch.to_pydict()
        texts = table[recipe["text_field"]]
        token_values = table.get(recipe.get("token_field", ""),
                                 [None] * len(texts))
        for text, token_value in zip(texts, token_values):
            result = maybe_write_row(recipe, file_meta, text, token_value,
                                     writers, manifest, source_meta, ordinal,
                                     source_budget)
            if result is None:
                return file_rows, file_tokens, ordinal
            token_estimate, ordinal = result
            if token_estimate == 0:
                continue
            file_rows += 1
            file_tokens += token_estimate
    return file_rows, file_tokens, ordinal


def maybe_write_row(recipe, file_meta, text, token_value, writers, manifest,
                    source_meta, ordinal, source_budget):
    if reached_budget(manifest, source_meta, source_budget):
        return None
    if not isinstance(text, str) or not text.strip():
        return 0, ordinal
    tokens = estimated_tokens(token_value, text)
    if (manifest["token_estimate"] + tokens > manifest["target_token_estimate"]
            or source_meta["token_estimate"] + tokens > source_budget):
        return None
    split = split_for_ordinal(ordinal)
    writers[split].write(row_for(recipe, file_meta, text, tokens, ordinal))
    source_meta["row_count"] += 1
    source_meta["token_estimate"] += tokens
    manifest["row_count"] += 1
    manifest["split_rows"][split] += 1
    manifest["token_estimate"] += tokens
    return tokens, ordinal + 1


def row_for(recipe, file_meta, text, tokens, ordinal):
    return {
        "id": f"public-pretrain-{recipe['config']}-{ordinal + 1:09d}",
        "mode": "pretrain",
        "language": recipe["language"],
        "domain": recipe["name"],
        "text": text,
        "metadata": {
            "provenance": "public-pretrain",
            "license": recipe["license"],
            "source_dataset": recipe["dataset"],
            "source_config": recipe["config"],
            "source_split": "train",
            "source_revision": recipe["revision"],
            "source_file": file_meta["filename"],
            "source_sha256": file_meta["sha256"],
            "field_policy": "text-only",
            "token_estimate": tokens,
        },
    }


def reached_budget(manifest, source_meta, source_budget):
    return (manifest["token_estimate"] >= manifest["target_token_estimate"]
            or source_meta["token_estimate"] >= source_budget)
