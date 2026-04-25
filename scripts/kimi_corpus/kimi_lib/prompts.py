from __future__ import annotations

import re
from pathlib import Path

from .records import now_iso


def render_prompt(config: dict, mode: str, documents: int, sample: bool) -> str:
    template = prompt_path(config, mode).read_text(encoding="utf-8")
    plan = domain_language_plan(config, mode, documents)
    replacements = {
        "MODE": mode,
        "DOCUMENTS": str(documents),
        "LANGUAGE_PLAN": plan["languages"],
        "DOMAIN_PLAN": plan["domains"],
        "DIFFICULTY_PLAN": plan["difficulties"],
        "PROMPT_VERSION": str(config.get("prompt_version", "v1")),
        "SAMPLE_NOTE": "This is a sample batch for quality inspection." if sample else "This is a production shard batch.",
        "GENERATED_AT": now_iso(),
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def prompt_path(config: dict, mode: str) -> Path:
    prompt_dir = Path(str(config.get("prompt_dir", "scripts/kimi_corpus/prompts")))
    version = str(config.get("prompt_version", "v1"))
    path = prompt_dir / f"{mode}_{version}.txt"
    return path if path.exists() else prompt_dir / f"{mode}_v1.txt"


def domain_language_plan(config: dict, mode: str, documents: int) -> dict:
    domains = config.get("pretrain_domains" if mode == "pretrain" else "sft_domains", [])
    return {
        "languages": ", ".join(["en", "ja", "mixed"][: min(3, documents)]),
        "domains": ", ".join(domains[: max(5, min(len(domains), documents))]),
        "difficulties": "introductory, intermediate, advanced",
    }


def extract_prompt_candidates(text: str) -> dict[str, str]:
    candidates = {}
    for name in ["pretrain", "sft"]:
        match = re.search(rf"<{name}_prompt>(.*?)</{name}_prompt>", text, re.DOTALL | re.IGNORECASE)
        if match:
            candidates[name] = match.group(1).strip()
    return candidates


def prompt_candidate_valid(body: str, name: str) -> bool:
    lower = body.lower()
    required = ["jsonl", "no prose", "copyright", "prompt_version"]
    required += ["text", "metadata", "pretrain"] if name == "pretrain" else ["messages", "assistant", "sft"]
    return all(item in lower for item in required)


def next_prompt_version(prompt_dir: Path) -> str:
    versions = []
    for path in prompt_dir.glob("*_v*.txt"):
        match = re.search(r"_v(\d+)\.txt$", path.name)
        if match:
            versions.append(int(match.group(1)))
    return f"v{max(versions, default=1) + 1}"
