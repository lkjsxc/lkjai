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


def artifact_text_checksum(text: str) -> str:
    h = 1469598103934665603
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return f"{h:x}"


def write_cache(root: Path) -> Path:
    cache = root / "train" / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    rows = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    flat = [token for row in rows for token in row]
    (cache / "metadata.json").write_text(json.dumps({
        "format": "lkjai-packed-cache-v2", "split": "train",
        "objective": "causal_lm_full", "sequence_len": 8,
        "vocab_size": 8192, "token_dtype": "uint16",
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
    assert report["decoder_status"] == "partial_cuda", report
    assert report["implementation_status"] == "partial_cuda", report
    assert report["accepted_cuda_training"] is False, report
    assert report["decoder_cuda_path"] is True, report
    assert report["decode_supported"] is True, report
    assert report["decoder_cuda_slice"] == "embedding_lm_head", report
    assert report["decoder_block_backend"] == "static_reference", report
    assert report["forward_backend"] == "cuda_bf16_embedding_lm_head", report
    assert report["backward_backend"] == "cuda_bf16_embedding_lm_head", report
    assert report["optimizer_backend"] == "cuda_adamw_fp32", report
    assert report["attention_backend"] == "not_implemented", report
    assert report["matmul_backend"] == "cublaslt", report
    assert "autoregressive_decode_unsupported" not in report["limitations"], report
    assert report["loss_finite"] is True, report
    assert report["weight_changed"] is True, report
    assert report["target_seconds"] == 60, report
    artifact = root / "train" / "exports" / "decoder-smoke"
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["kind"] == "decoder", manifest
    source_tokenizer = Path(__file__).resolve().parents[2] / "data" / "train" / "tokenizer" / "tokenizer.json"
    tokenizer = json.loads((artifact / "tokenizer.json").read_text())
    assert tokenizer == json.loads(source_tokenizer.read_text())
    assert tokenizer["model"]["type"] == "BPE", tokenizer
    assert tokenizer["pre_tokenizer"]["type"] == "ByteLevel", tokenizer
    subprocess.run([inspect_bin, "--model-dir", str(artifact)], check=True)
    bad_checksum = root / "bad-tokenizer-checksum"
    shutil.copytree(artifact, bad_checksum)
    bad_manifest = json.loads((bad_checksum / "manifest.json").read_text())
    bad_manifest["tokenizer_checksum"] = "bad"
    (bad_checksum / "manifest.json").write_text(json.dumps(bad_manifest) + "\n")
    bad = subprocess.run([inspect_bin, "--model-dir", str(bad_checksum)],
                         text=True, capture_output=True)
    assert bad.returncode != 0 and "tokenizer_checksum" in bad.stderr, bad.stderr
    missing_tag = root / "missing-atomic-tag"
    shutil.copytree(artifact, missing_tag)
    tokenizer_text = (missing_tag / "tokenizer.json").read_text().replace(
        "<tool_name>", "<tool-name>", 1)
    (missing_tag / "tokenizer.json").write_text(tokenizer_text)
    tag_manifest = json.loads((missing_tag / "manifest.json").read_text())
    tag_manifest["tokenizer_checksum"] = artifact_text_checksum(tokenizer_text)
    (missing_tag / "manifest.json").write_text(json.dumps(tag_manifest) + "\n")
    bad = subprocess.run([inspect_bin, "--model-dir", str(missing_tag)],
                         text=True, capture_output=True)
    assert bad.returncode != 0 and "atomic tag" in bad.stderr, bad.stderr
    logits = subprocess.run([logits_bin, "--model-dir", str(artifact),
                             "--tokens", "1,2,3"], text=True,
                            capture_output=True, check=True)
    assert json.loads(logits.stdout)["shape"] == [1, 8192]
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
        assert payload["choices"][0]["finish_reason"] in {"stop", "length"}, payload
        assert payload["choices"][0]["lkjai_stop_reason"] in {
            "eos", "end_action", "max_tokens"}, payload
        body["top_p"] = 0.0
        status, text = request(port, "POST", "/v1/chat/completions", body)
        assert status == 400 and "choices" not in text, text
        body["top_p"] = 1.0
        body["max_tokens"] = 513
        status, text = request(port, "POST", "/v1/chat/completions", body)
        assert status == 400 and "choices" not in text, text
        bad_role = {"model": "decoder-smoke",
                    "messages": [{"role": "invalid", "content": "x"}]}
        status, text = request(port, "POST", "/v1/chat/completions", bad_role)
        assert status == 400 and "choices" not in text, text
        wrong_model = {"model": "not-loaded",
                       "messages": [{"role": "user", "content": "x"}]}
        status, text = request(port, "POST", "/v1/chat/completions", wrong_model)
        assert status == 404 and "choices" not in text, text
    finally:
        server.terminate()
        server.wait(timeout=5)
    shutil.rmtree(root)


if __name__ == "__main__":
    main()
