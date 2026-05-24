import json
import re
import socket
import urllib.error
import urllib.request

from .cli_runner import call_kimi_cli_jobs, kimi_cli_job
from .common import ApiResult
from .constants import DEFAULT_API_BASE, DEFAULT_MODEL


def call_kimi(api_key, config, messages):
    if str(config.get("api_provider", "")) == "kimi-cli":
        return call_kimi_cli(api_key, config, messages)
    base = str(config.get("api_base_url", DEFAULT_API_BASE)).rstrip("/")
    payload = {
        "model": str(config.get("api_model", DEFAULT_MODEL)),
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": int(config.get("max_response_tokens", 12000)),
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": str(config.get("user_agent", "lkjai-kimi-sft/1.0")),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=int(config.get("timeout_seconds", 240))
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return classify_http_error(exc)
    except (TimeoutError, socket.timeout) as exc:
        return ApiResult("retry", error=f"Kimi API timeout: {exc}", retryable=True)
    except urllib.error.URLError as exc:
        return ApiResult("retry", error=f"Kimi API connection error: {exc.reason}", retryable=True)
    try:
        return ApiResult("pass", text=body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ApiResult("fail", error="Kimi API response did not contain choices[0].message.content")


def classify_http_error(exc):
    safe = exc.read(2048).decode("utf-8", "replace")
    status = exc.code
    if status == 401:
        return ApiResult("fail", error="Kimi API returned 401 unauthorized", unauthorized=True)
    if status == 429:
        return ApiResult("quota", error="Kimi API returned 429 rate/quota limit", quota_exhausted=True)
    if status == 403 and "access_terminated" in safe:
        return ApiResult(
            "fail",
            error="Kimi API returned 403 access_terminated",
            access_terminated=True,
        )
    if status >= 500:
        return ApiResult("retry", error=f"Kimi API returned HTTP {status}: {safe[:160]}", retryable=True)
    return ApiResult("fail", error=f"Kimi API returned HTTP {status}: {safe[:160]}")


def parse_kimi_rows(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return value["rows"]
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and {"messages", "tags", "meta"} <= set(value):
        return [value]
    raise ValueError("Kimi JSON did not contain rows")


def call_kimi_cli(api_key, config, messages):
    result = call_kimi_cli_jobs(api_key, config, [kimi_cli_job("job-0", 0, "", "", messages)])
    return result[0] if result else ApiResult("fail", error="Kimi CLI runner returned no result")
