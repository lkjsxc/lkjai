#!/usr/bin/env python3
import json
import subprocess
import sys


FIELDS = [
    "cuda_driver_version",
    "cuda_runtime_version",
    "cudnn_version",
    "cuda_device_count",
    "cuda_device_index",
    "cuda_total_global_memory",
    "cuda_sm_count",
    "cuda_arch_flags",
    "cuda_arch_source",
    "async_alloc_supported",
]


def main() -> None:
    result = subprocess.run([sys.argv[1]], text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass", payload
    for field in FIELDS:
        assert field in payload, payload
    assert payload["cuda_device_count"] >= 1, payload
    assert payload["cuda_total_global_memory"] > 0, payload
    assert payload["cuda_sm_count"] > 0, payload
    assert payload["cuda_arch_flags"], payload


if __name__ == "__main__":
    main()
