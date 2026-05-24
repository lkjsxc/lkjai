import json
import os
import subprocess

from .common import ApiResult
from .constants import DEFAULT_API_BASE, DEFAULT_MODEL


def kimi_cli_job(job_id, ordinal, split, family, messages, constraints=None):
    return {
        "job_id": job_id,
        "ordinal": ordinal,
        "split": split,
        "template_family": family,
        "messages": messages,
        "constraints": constraints or {},
        "input_jsonl": "\n".join(
            json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            for message in messages
        )
        + "\n",
    }


def call_kimi_cli_jobs(api_key, config, jobs):
    runner = os.environ.get("KIMI_CLI_RUNNER") or str(
        config.get(
            "kimi_cli_runner",
            config.get("kimi_cli_bridge", "/tmp/lkjai-native-kimi-cli-runner"),
        )
    )
    env = dict(os.environ)
    env["KIMI_API_KEY"] = api_key
    command = [
        runner,
        "--base-url",
        str(config.get("api_base_url", DEFAULT_API_BASE)),
        "--model",
        str(config.get("api_model", DEFAULT_MODEL)),
        "--max-steps",
        str(config.get("kimi_cli_max_steps", 3)),
        "--max-tokens",
        str(config.get("max_response_tokens", 12000)),
        "--parallelism",
        str(max(1, int(config.get("parallelism", 1)))),
        "--max-retries",
        str(max(0, int(config.get("max_retries", 0)))),
    ]
    payload = "\n".join(
        json.dumps(job, ensure_ascii=False, separators=(",", ":")) for job in jobs
    )
    if payload:
        payload += "\n"
    try:
        result = subprocess.run(
            command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=runner_timeout(config, len(jobs)),
            env=env,
            check=False,
        )
    except FileNotFoundError:
        return [ApiResult("fail", error=f"Kimi CLI runner not found: {runner}") for _ in jobs]
    except subprocess.TimeoutExpired:
        return [ApiResult("retry", error="Kimi CLI runner timeout", retryable=True) for _ in jobs]
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[0][:160]}" if detail else ""
        return [
            ApiResult(
                "retry",
                error=f"Kimi CLI runner failed: exit {result.returncode}{suffix}",
                retryable=True,
            )
            for _ in jobs
        ]
    return parse_runner_results(result.stdout, jobs)


def runner_timeout(config, job_count):
    per_job = int(config.get("timeout_seconds", 240))
    parallelism = max(1, int(config.get("parallelism", 1)))
    waves = max(1, (max(1, job_count) + parallelism - 1) // parallelism)
    return per_job * waves + 30


def parse_runner_results(text, jobs):
    by_id = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        job_id = payload.get("job_id")
        if isinstance(job_id, str):
            by_id[job_id] = payload
    return [runner_result(job, by_id.get(job["job_id"])) for job in jobs]


def runner_result(job, payload):
    if payload is None:
        return ApiResult("retry", error="Kimi CLI runner omitted job result", retryable=True)
    status = str(payload.get("status", "fail"))
    error = str(payload.get("error", ""))
    return ApiResult(
        "pass" if status == "pass" else status,
        text=str(payload.get("text", "")),
        error=error,
        attempts=int(payload.get("attempts", 0) or 0),
        elapsed_ms=int(payload.get("elapsed_ms", 0) or 0),
        retryable=status == "retry",
        quota_exhausted=status == "quota",
        unauthorized="unauthoriz" in error.lower(),
        access_terminated="access_terminated" in error,
    )
