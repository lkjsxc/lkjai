#!/usr/bin/env python3
import http.client
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


def write_cache(root: Path) -> Path:
    cache = root / "train" / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    rows = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    flat = [token for row in rows for token in row]
    (cache / "metadata.json").write_text(json.dumps({
        "format": "lkjai-packed-cache-v2", "split": "train",
        "objective": "causal_lm_full", "sequence_len": 8,
        "vocab_size": 256, "token_dtype": "uint16",
        "row_count": len(rows), "token_count": len(flat)}))
    (cache / "tokens.bin").write_bytes(struct.pack("<" + "H" * len(flat), *flat))
    (cache / "loss_mask.bin").write_bytes(bytes([1] * len(flat)))
    (cache / "starts.bin").write_bytes(struct.pack("<2Q", 0, 8))
    return cache


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload = None if body is None else json.dumps(body)
    headers = {"content-type": "application/json"} if payload else {}
    conn.request(method, path, payload, headers)
    response = conn.getresponse()
    text = response.read().decode("utf-8")
    conn.close()
    return response.status, text


def wait_ready(port):
    for _ in range(50):
        try:
            status, _ = request(port, "GET", "/v1/models")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("decoder server did not become ready")


def main():
    train_bin, logits_bin, inspect_bin, server_bin = sys.argv[1:5]
    root = Path("/tmp/lkjai-decoder-contract")
    if root.exists():
        shutil.rmtree(root)
    cache = write_cache(root)
    env = {**os.environ.copy(), "DATA_DIR": str(root / "train"),
           "MODEL_NAME": "decoder-smoke", "TRAIN_TARGET_SECONDS": "60"}
    cfg = Path(__file__).resolve().parents[2] / "configs" / "native" / "decoder_debug_bf16.json"
    cmd = [train_bin, "--train", "--mode", "decoder", "--config", str(cfg),
           "--packed-cache", str(cache), "--seq-len", "8", "--max-steps", "2",
           "--lr", "0.01"]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report["model_kind"] == "decoder", report
    assert report["decoder_status"] == "experimental", report
    assert report["target_seconds"] == 60, report
    artifact = root / "train" / "exports" / "decoder-smoke"
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["kind"] == "decoder", manifest
    subprocess.run([inspect_bin, "--model-dir", str(artifact)], check=True)
    logits = subprocess.run([logits_bin, "--model-dir", str(artifact),
                             "--tokens", "1,2,3"], text=True,
                            capture_output=True, check=True)
    assert json.loads(logits.stdout)["shape"] == [1, 256]
    port = 18082
    server = subprocess.Popen([server_bin], env={**os.environ.copy(),
        "MODEL_ROOT": str(root / "models"), "MODEL_NAME": "decoder-smoke",
        "INFERENCE_HOST": "127.0.0.1", "INFERENCE_PORT": str(port)})
    try:
        wait_ready(port)
        body = {"model": "decoder-smoke",
                "messages": [{"role": "user", "content": "smoke"}],
                "max_tokens": 2, "temperature": 0.0}
        status, text = request(port, "POST", "/v1/chat/completions", body)
        assert status == 200, text
        payload = json.loads(text)
        assert payload["choices"][0]["message"]["content"], payload
    finally:
        server.terminate()
        server.wait(timeout=5)
    shutil.rmtree(root)


if __name__ == "__main__":
    main()
