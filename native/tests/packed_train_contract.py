#!/usr/bin/env python3
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
CAPABILITY_FIELDS = ("cuda_driver_version", "cuda_runtime_version", "cudnn_version", "cuda_device_count", "cuda_device_index", "cuda_total_global_memory", "cuda_sm_count", "cuda_arch_flags")
def assert_capability_fields(report: dict):
    for fields in [report, report["capability"]]:
        for key in CAPABILITY_FIELDS:
            assert key in fields, fields
def write_cache(root: Path, version: str = "lkjai-packed-cache-v2", vocab_size: int = 256):
    cache = root / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    meta = {
        "format": version,
        "split": "train",
        "objective": "causal_lm_full",
        "sequence_len": 8,
        "seq_len": 8,
        "vocab_size": vocab_size,
        "token_dtype": "uint16",
        "row_count": 1,
        "token_count": 8,
    }
    (cache / "metadata.json").write_text(json.dumps(meta))
    (cache / "tokens.bin").write_bytes(struct.pack("<8H", *range(8)))
    (cache / "loss_mask.bin").write_bytes(bytes([1] * 8))
    (cache / "starts.bin").write_bytes(struct.pack("<Q", 0))
def run_train(train_bin: str, root: Path, repo: Path, steps: int, extra=None):
    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(root),
            "MODEL_NAME": "packed-smoke",
            "TRAIN_MAX_OPTIMIZER_STEPS": str(steps),
            "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": "1",
        }
    )
    cmd = [
        train_bin,
        "--train",
        "--config",
        str(repo / "configs" / "native" / "native_debug_bf16.json"),
        "--seq-len",
        "8",
        "--max-steps",
        str(steps),
    ]
    if extra:
        cmd.extend(extra)
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    report_path = root / "runs" / "train-report.json"
    assert report_path.is_file(), result.stdout
    persisted = json.loads(report_path.read_text())
    assert persisted["schema_version"] == payload["schema_version"] == 3
    assert persisted["trainer_mode"] == payload["trainer_mode"] == "train"
    assert persisted["run_purpose"] == payload["run_purpose"] == "accepted_training"
    expected = {"precision_mode": "fp32-master-bf16-shadow-bf16-export", "master_dtype": "f32",
                "shadow_dtype": "bf16", "accumulation_dtype": "f32", "export_dtype": "bf16"}
    for key, expected in expected.items():
        assert persisted[key] == expected
    assert persisted["dense_cuda_path"] is True
    expected = {"loader_backend": "persistent_packed_cache_reader", "row_layout": "dense_physical_bxseq_masked_final_token",
                "matmul_plan_cache_enabled": True, "buffer_reuse_enabled": True, "timing_source": "cuda_events_with_boundary_sync",
                "forward_backend": "cuda_bf16_cublaslt", "backward_backend": "cuda_bf16_cublaslt_scatter",
                "backward_gemm_enabled": True, "embedding_grad_backend": "token_scatter_add_fp32", "optimizer_backend": "cuda_adamw_fp32",
                "loss_kernel_backend": "block_row_softmax_fp32", "loss_readback_mode": "optimizer_step_deferred_pinned",
                "logits_readback_mode": "single_row_capture", "dense_stream_count": 2, "dense_batch_slot_count": 3,
                "copy_compute_overlap_enabled": True, "batch_staging_backend": "triple_slot_pinned_direct_read"}
    for key, expected in expected.items():
        assert persisted[key] == expected
    assert persisted["accepted_cuda_training"] is True
    assert persisted["implementation_status"] == "accepted"
    assert persisted["dense_step_logits_bytes"] == persisted["dense_step_grad_logits_bytes"]
    assert persisted["dense_logits_readback_bytes"] == 256 * 4
    assert persisted["dense_step_d_hidden_bytes"] > 0 and persisted["cublaslt_workspace_bytes"] > 0
    assert persisted["cuda_probe_passed"] is True
    assert_capability_fields(persisted)
    assert persisted["logits_check"]["validation_target"] == "exported_bf16_weights"
    assert persisted["logits_check"]["status"] == "pass"
    assert persisted["logits_check"]["reference_check"] == "pass"
    assert persisted["logits_check"]["max_abs_diff"] <= persisted["logits_check"]["tolerance"]
    return payload
def check_schema(artifact: Path, inspect_bin: str):
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["format"] == "lkjai-native-artifact-v2"
    assert manifest["kind"] == "dense"
    assert manifest["artifact_kind"] == "export"
    assert manifest["weights_checksum"]
    assert manifest["config_checksum"]
    assert manifest["tokenizer_checksum"]
    index = json.loads((artifact / "weights.index.json").read_text())
    names = {tensor["name"] for tensor in index["tensors"]}
    assert {"tok_embeddings", "lm_head"} <= names
    first = index["tensors"][0]
    assert {"name", "dtype", "shape", "byte_offset", "byte_length"} <= set(first)
    checkpoint = artifact.parents[1] / "checkpoints" / "latest"
    ckpt_manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert ckpt_manifest["kind"] == "dense"
    assert ckpt_manifest["artifact_kind"] == "checkpoint"
    assert (checkpoint / "optimizer.index.json").is_file()
    opt_index = json.loads((checkpoint / "optimizer.index.json").read_text())
    opt_names = {tensor["name"] for tensor in opt_index["tensors"]}
    assert {"master.tok_embeddings", "adam_m.tok_embeddings",
            "adam_v.tok_embeddings", "master.lm_head",
            "adam_m.lm_head", "adam_v.lm_head"} <= opt_names
    broken = artifact.parent / "broken-schema"
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(artifact, broken)
    bad = dict(manifest)
    bad["tokenizer_checksum"] = "bad"
    (broken / "manifest.json").write_text(json.dumps(bad))
    result = subprocess.run([inspect_bin, "--model-dir", str(broken)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "checksum" in result.stderr
def main():
    train_bin, logits_bin, inspect_bin = sys.argv[1:4]
    repo = Path(__file__).resolve().parents[2]
    root = Path("/tmp/lkjai-packed-train")
    shutil.rmtree(root, ignore_errors=True)
    write_cache(root)
    payload = run_train(train_bin, root, repo, 2)
    assert payload["status"] == "success" and payload["dense_cuda_path"] is True
    assert "transformer_path" not in payload, payload
    assert payload["initial_loss"] > payload["loss"], payload
    assert payload["loss_finite"] is True, payload
    assert payload["weight_changed"] is True, payload
    assert payload["logits_checksum"], payload
    for key in ["batch_load", "h2d", "forward", "backward", "optimizer", "checkpoint", "export"]:
        assert key in payload["timings"], payload
        assert payload["timings"][key] >= 0, payload
    manifest = root / "exports" / "packed-smoke" / "manifest.json"
    assert "lkjai-native-artifact-v2" in manifest.read_text()
    artifact = root / "exports" / "packed-smoke"
    subprocess.run([inspect_bin, "--model-dir", str(artifact)], check=True)
    check_schema(artifact, inspect_bin)
    result = subprocess.run([logits_bin, "--model-dir", str(artifact), "--tokens", "1,2,3"],
                            text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["finite"] is True, result.stdout
    assert payload["shape"] == [1, 256], result.stdout
    assert payload["reference_check"] == "not_requested", result.stdout
    first_logits_checksum = payload["checksum"]
    result = subprocess.run([logits_bin, "--model-dir", str(artifact),
                             "--tokens", "1,2,3", "--reference-checkpoint",
                             str(root / "checkpoints" / "latest")],
                            text=True, capture_output=True, check=True)
    ref_payload = json.loads(result.stdout)
    assert ref_payload["reference_check"] == "pass", result.stdout
    assert ref_payload["max_abs_diff"] <= ref_payload["tolerance"], result.stdout
    assert ref_payload["mean_abs_diff"] <= ref_payload["tolerance"], result.stdout
    result = subprocess.run([logits_bin, "--model-dir", str(artifact), "--tokens", "1,2,3"],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["checksum"] == first_logits_checksum
    payload = run_train(train_bin, root, repo, 1,
                        ["--resume", str(root / "checkpoints" / "latest")])
    assert payload["start_step"] == 2, result.stdout
    assert payload["steps"] == 3, result.stdout
    mono = root.parent / "lkjai-packed-train-mono"
    shutil.rmtree(mono, ignore_errors=True)
    write_cache(mono)
    mono_payload = run_train(train_bin, mono, repo, 3)
    assert payload["export_checksum"] == mono_payload["export_checksum"]
    assert payload["logits_checksum"] == mono_payload["logits_checksum"]
    repeated = root.parent / "lkjai-packed-train-repeat"
    shutil.rmtree(repeated, ignore_errors=True)
    write_cache(repeated)
    repeated_payload = run_train(train_bin, repeated, repo, 2)
    assert repeated_payload["export_checksum"]
    assert repeated_payload["logits_checksum"] == first_logits_checksum
    bad = subprocess.run(
        [train_bin, "--train", "--config",
         str(repo / "configs" / "native" / "native_40m_bf16.json"),
         "--seq-len", "8", "--max-steps", "1", "--resume",
         str(root / "checkpoints" / "latest")],
        env={**os.environ.copy(), "DATA_DIR": str(root)},
        text=True, capture_output=True)
    assert bad.returncode != 0
    assert "mismatch" in bad.stderr
    incompatible = root.parent / "lkjai-packed-train-incompatible-cache"
    shutil.rmtree(incompatible, ignore_errors=True)
    write_cache(incompatible, vocab_size=300)
    bad_cache = subprocess.run(
        [train_bin, "--train", "--config",
         str(repo / "configs" / "native" / "native_debug_bf16.json"),
         "--packed-cache",
         str(incompatible / "datasets" / "packed" / "train-causal_lm_full-seq1024"),
         "--seq-len", "8", "--max-steps", "1"],
        env={**os.environ.copy(), "DATA_DIR": str(incompatible)},
        text=True, capture_output=True)
    assert bad_cache.returncode != 0
    assert "packed cache vocab_size exceeds" in bad_cache.stderr
if __name__ == "__main__":
    main()
