from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .kimi_cli import KimiResult
from .kimi_keys import fingerprint, load_api_keys, redact


TRANSIENT_TYPES = {
    "engine_overloaded_error",
    "rate_limit_reached_error",
    "tokens_per_minute_reached_error",
    "requests_per_minute_reached_error",
    "concurrency_reached_error",
    "server_error",
}


class KimiApiRunner:
    def __init__(self, logs_dir: Path, config: dict, args):
        self.logs_dir = logs_dir
        self.base_url = str(getattr(args, "api_base_url", None) or config.get("api_base_url"))
        self.base_url = self.base_url.rstrip("/") or "https://api.moonshot.ai/v1"
        self.model = str(getattr(args, "api_model", None) or config.get("api_model") or "kimi-k2.6")
        self.keys = load_api_keys(getattr(args, "api_key_file", ""))
        self.next_key = 0
        self.variant = "kimi_api"
        self.executable = f"{self.base_url}/chat/completions"

    def invoke(self, prompt: str, call_id: str, timeout_seconds: int, max_retries: int) -> KimiResult:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.logs_dir / f"{call_id}.prompt.txt"
        stdout_path = self.logs_dir / f"{call_id}.stdout.log"
        stderr_path = self.logs_dir / f"{call_id}.stderr.log"
        prompt_path.write_text(redact(prompt, self.keys), encoding="utf-8")
        started = time.perf_counter()
        if not self.keys:
            stderr_path.write_text("MOONSHOT_API_KEY or --api-key-file is required\n", encoding="utf-8")
            return KimiResult(1, stdout_path, stderr_path, self.variant, time.perf_counter() - started)
        for attempt in range(max_retries + 1):
            key = self._key()
            try:
                rows = self._chat(prompt, key, timeout_seconds)
                stdout_path.write_text(rows, encoding="utf-8")
                stderr_path.write_text(f"key={fingerprint(key)} model={self.model}\n", encoding="utf-8")
                return KimiResult(0, stdout_path, stderr_path, self.variant, time.perf_counter() - started)
            except ApiError as error:
                stderr_path.write_text(redact(error.message, self.keys), encoding="utf-8")
                if not error.transient or attempt >= max_retries:
                    return KimiResult(error.exit_code, stdout_path, stderr_path, self.variant, time.perf_counter() - started)
                time.sleep(2**attempt)
        return KimiResult(1, stdout_path, stderr_path, self.variant, time.perf_counter() - started)

    def _key(self) -> str:
        key = self.keys[self.next_key % len(self.keys)]
        self.next_key += 1
        return key

    def _chat(self, prompt: str, key: str, timeout_seconds: int) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "response_format": row_schema(),
            "max_completion_tokens": 8192,
            "prompt_cache_key": f"lkjai:{self.model}:sft",
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        data = self._post("/chat/completions", body, key, timeout_seconds)
        content = data["choices"][0]["message"]["content"]
        return rows_to_jsonl(content)

    def _post(self, path: str, body: dict, key: str, timeout_seconds: int) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            raise ApiError.from_http(error.code, text) from error
        except Exception as error:
            raise ApiError(str(error), True, 75) from error


class ApiError(Exception):
    def __init__(self, message: str, transient: bool, exit_code: int):
        super().__init__(message)
        self.message = message
        self.transient = transient
        self.exit_code = exit_code

    @classmethod
    def from_http(cls, status: int, text: str) -> "ApiError":
        error_type = ""
        try:
            error_type = json.loads(text).get("error", {}).get("type", "")
        except json.JSONDecodeError:
            pass
        transient = status >= 500 or status == 429 or error_type in TRANSIENT_TYPES
        return cls(f"http_status={status} error_type={error_type} body={text}", transient, 75 if transient else 1)


def rows_to_jsonl(content: str) -> str:
    try:
        parsed = json.loads(content)
        rows = parsed.get("rows") or parsed.get("documents") or []
        if isinstance(rows, list):
            return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    except json.JSONDecodeError:
        pass
    return content


def system_prompt() -> str:
    return "Return JSON only. Produce {\"rows\":[...]} with lkjai SFT row objects."


def row_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "lkjai_kimi_rows",
            "schema": {
                "type": "object",
                "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
                "required": ["rows"],
                "additionalProperties": False,
            },
        },
    }
