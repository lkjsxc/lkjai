#!/usr/bin/env python3
import http.client
import json
import os
import subprocess
import sys
import time


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    payload = None if body is None else json.dumps(body)
    headers = {"content-type": "application/json"} if payload else {}
    conn.request(method, path, payload, headers)
    response = conn.getresponse()
    text = response.read().decode("utf-8")
    conn.close()
    return response.status, text


def assert_capability(payload):
    for key in [
        "bf16_supported",
        "async_alloc_supported",
        "cuda_driver_version",
        "cuda_device_count",
        "cuda_device_index",
        "cuda_total_global_memory",
        "cuda_sm_count",
        "cuda_arch_flags",
    ]:
        assert key in payload, payload


def wait_ready(port):
    for _ in range(50):
        try:
            status, text = request(port, "GET", "/healthz")
            if status == 200:
                assert_capability(json.loads(text))
            status, text = request(port, "GET", "/v1/models")
            if status == 200:
                payload = json.loads(text)
                assert_capability(payload)
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("native server did not become ready")


def main():
    server, model_root, model_name = sys.argv[1:4]
    port = 18081
    env = os.environ.copy()
    env.update(
        {
            "MODEL_ROOT": model_root,
            "MODEL_NAME": model_name,
            "INFERENCE_HOST": "127.0.0.1",
            "INFERENCE_PORT": str(port),
        }
    )
    process = subprocess.Popen(
        [server], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        wait_ready(port)
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": "smoke"}],
            "max_tokens": 512,
            "temperature": 0.2,
        }
        status, text = request(port, "POST", "/v1/chat/completions", body)
        assert status == 422, text
        assert "unsupported" in text, text

        body["max_tokens"] = 1
        status, text = request(port, "POST", "/v1/chat/completions", body)
        assert status == 422, text
        assert "choices" not in text, text
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


if __name__ == "__main__":
    main()
