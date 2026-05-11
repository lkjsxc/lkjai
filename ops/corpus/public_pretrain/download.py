import json
from urllib.parse import urlencode

import requests

from .auth import auth_headers
from .common import (
    DEFAULT_RAW,
    DEFAULT_SOURCE,
    PARQUET_API,
    env_path,
    file_sha256,
    load_recipes,
    safe_command_meta,
    write_json,
)


def parquet_api(dataset, config, revision):
    query = urlencode({
        "dataset": dataset,
        "config": config,
        "split": "train",
        "revision": revision,
    })
    response = requests.get(f"{PARQUET_API}?{query}", headers=auth_headers(),
                            timeout=60)
    response.raise_for_status()
    files = response.json().get("parquet_files", [])
    if not files:
        raise RuntimeError(f"no parquet files for {dataset}/{config}")
    return files


def download_file(url, path, expected_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and expected_size and path.stat().st_size == expected_size:
        return "present"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, headers=auth_headers(), stream=True,
                      timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)
    return "downloaded"


def download(_args):
    source_file = env_path("CORPUS_SOURCE_FILE", DEFAULT_SOURCE)
    raw_dir = env_path("TRAIN_PUBLIC_DATA_DIR", DEFAULT_RAW)
    manifest = {
        "schema": "lkjai-public-pretrain-download",
        "command": safe_command_meta("download-public-pretrain"),
        "sources": [],
    }
    for recipe in load_recipes(source_file):
        source = source_manifest(recipe)
        for item in parquet_api(recipe["dataset"], recipe["config"],
                                recipe["revision"]):
            local = raw_dir / recipe["config"] / "train" / item["filename"]
            status = download_file(item["url"], local, item.get("size", 0))
            source["files"].append({
                "filename": item["filename"],
                "url": item["url"],
                "bytes": local.stat().st_size,
                "sha256": file_sha256(local),
                "status": status,
            })
            print(f"{recipe['config']}/{item['filename']}: {status}",
                  flush=True)
        manifest["sources"].append(source)
    write_json(raw_dir / "download-manifest.json", manifest)
    print(json.dumps({"status": "pass",
                      "manifest": str(raw_dir / "download-manifest.json")}))


def source_manifest(recipe):
    return {
        "name": recipe["name"],
        "dataset": recipe["dataset"],
        "config": recipe["config"],
        "dataset_revision": recipe["revision"],
        "split": "train",
        "license": recipe["license"],
        "text_field": recipe["text_field"],
        "token_field": recipe.get("token_field", ""),
        "files": [],
    }
