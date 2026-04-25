from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


TRANSIENT_EXIT_CODES = {75}


@dataclass
class KimiResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    command_variant: str
    duration_seconds: float


class KimiRunner:
    def __init__(self, logs_dir: Path, fake_kimi: str = ""):
        self.logs_dir = logs_dir
        self.executable = str(Path(fake_kimi).resolve()) if fake_kimi else discover_kimi()
        self.help_text = kimi_help(self.executable)
        self.variant = choose_variant(self.help_text)

    def invoke(self, prompt: str, call_id: str, timeout_seconds: int, max_retries: int) -> KimiResult:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.logs_dir / f"{call_id}.prompt.txt"
        stdout_path = self.logs_dir / f"{call_id}.stdout.log"
        stderr_path = self.logs_dir / f"{call_id}.stderr.log"
        prompt_path.write_text(prompt, encoding="utf-8")
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            command, stdin = self.command(prompt)
            with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
                proc = None
                try:
                    proc = subprocess.Popen(command, cwd=self.logs_dir, stdin=subprocess.PIPE if stdin is not None else None, text=True, stdout=out, stderr=err, start_new_session=True)
                    proc.communicate(input=stdin, timeout=timeout_seconds)
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    if proc is not None:
                        terminate_group(proc)
                    err.write(f"kimi invocation timed out after {timeout_seconds} seconds\n")
                    returncode = 75
            result = KimiResult(returncode, stdout_path, stderr_path, self.variant, time.perf_counter() - started)
            if returncode == 0 or not is_transient_result(result) or attempt >= max_retries:
                return result
            time.sleep(2**attempt)
        return KimiResult(1, stdout_path, stderr_path, self.variant, 0.0)

    def command(self, prompt: str) -> tuple[list[str], str | None]:
        if self.variant == "quiet_prompt":
            return [self.executable, "--quiet", "-p", prompt], None
        if self.variant == "print_final_prompt":
            return [self.executable, "--print", "--output-format", "text", "--final-message-only", "-p", prompt], None
        return [self.executable, "--print", "--output-format", "text", "--final-message-only"], prompt


def discover_kimi() -> str:
    executable = shutil.which("kimi")
    if not executable:
        raise RuntimeError("kimi executable not found on PATH")
    return executable


def kimi_help(executable: str) -> str:
    proc = subprocess.run([executable, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    return proc.stdout


def choose_variant(help_text: str) -> str:
    if "--quiet" in help_text and "-p" in help_text:
        return "quiet_prompt"
    if "--print" in help_text and "--final-message" in help_text and "-p" in help_text:
        return "print_final_prompt"
    return "stdin_print_final"


def terminate_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def is_transient_result(result: KimiResult) -> bool:
    if result.returncode in TRANSIENT_EXIT_CODES:
        return True
    text = ""
    for path in [result.stderr_path, result.stdout_path]:
        if path.exists():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")[:4000].lower()
    markers = ("rate limit", "temporarily unavailable", "timeout", "timed out", "try again", "overloaded", "too many requests")
    return result.returncode != 0 and any(marker in text for marker in markers)
