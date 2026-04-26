from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .config import DEFAULT_RUN_DIR
from .kimi_cli import KimiRunner, is_transient_result
from .manifest import Manifest, split_for_shard
from .prompts import extract_prompt_candidates, next_prompt_version, prompt_candidate_valid, render_prompt
from .records import normalize_record, parse_jsonl_payload, sample_excerpts, write_jsonl_atomic
from .sample_report import sample_section
from .score import load_tokenizer, score_paths


class CorpusGenerator:
    def __init__(self, config: dict, args):
        self.config, self.args = config, args
        self.output_dir = Path(config.get("output_dir", "data/kimi_synthetic"))
        self.run_dir = Path(args.run_dir)
        if getattr(args, "stop_file", None) is None and str(config.get("stop_file", "")) == "runs/kimi_corpus/STOP":
            self.config["stop_file"] = str(self.run_dir / "STOP")
        self.logs_dir = self.run_dir / "logs"
        self.prompt_dir = Path(config.get("prompt_dir", "tools/kimi-corpus/prompts"))
        self.kimi = KimiRunner(self.logs_dir, args.fake_kimi)
        self.manifest = Manifest(self.output_dir, self.kimi.variant)
        self.tokenizer = self.load_optional_tokenizer()
        self.started = time.perf_counter()
        self.calls_started = 0
        self.pending_counts = {"pretrain": 0, "sft": 0}

    def run(self) -> None:
        self.prepare_dirs()
        if self.args.dry_run:
            print(json.dumps({"event": "dry_run", "config": self.config, "kimi": self.kimi.executable}))
            return
        with ThreadPoolExecutor(max_workers=int(self.config.get("parallelism", 1))) as pool:
            futures = {}
            while futures or not self.target_reached():
                while not self.target_reached() and len(futures) < int(self.config.get("parallelism", 1)):
                    if self.stop_requested() or self.max_calls_reached():
                        break
                    self.calls_started += 1
                    mode = self.choose_mode()
                    self.pending_counts[mode] += 1
                    futures[pool.submit(self.generate_one_shard, mode, False)] = mode
                if not futures:
                    break
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    mode = futures.pop(future)
                    self.pending_counts[mode] -= 1
                    future.result()
                    self.print_progress()
                    if float(self.config.get("sleep_between_calls", 0)) > 0:
                        time.sleep(float(self.config["sleep_between_calls"]))

    def sample_first_workflow(self) -> None:
        self.prepare_dirs()
        samples = self.output_dir / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        sections = ["# Kimi Sample Report", ""]
        try:
            docs = int(self.config.get("sample_documents", 20))
            version = str(self.config.get("prompt_version", "v1"))
            first = [self.generate_sample("pretrain", samples / f"pretrain_{version}.jsonl", docs), self.generate_sample("sft", samples / f"sft_{version}.jsonl", docs)]
            sections += [sample_section(version, score_paths(first, self.tokenizer)), sample_excerpts(first)]
            refined = self.refine_prompts("\n".join(sections))
            if refined:
                self.config["prompt_version"] = refined
                second = [self.generate_sample("pretrain", samples / f"pretrain_{refined}.jsonl", docs), self.generate_sample("sft", samples / f"sft_{refined}.jsonl", docs)]
                sections += [sample_section(refined, score_paths(second, self.tokenizer)), sample_excerpts(second)]
        except Exception as error:
            sections += ["## Workflow Blocker", "", f"- `{type(error).__name__}`: `{str(error)}`", ""]
            print(json.dumps({"event": "sample_first_blocked", "error": str(error)}), flush=True)
        report = self.run_dir / "sample_report.md"
        cmd = "`bash tools/kimi-corpus/launch_background.sh --config configs/corpus/kimi_500m.yaml --target-tokens 500000000 --parallelism 2 --output-dir corpus/generated/kimi-full-v1 --full`"
        sections += ["## Background Launch Command", "", cmd, ""]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(sections) + "\n", encoding="utf-8")
        print(json.dumps({"event": "sample_report", "path": str(report)}), flush=True)

    def generate_sample(self, mode: str, target: Path, docs: int) -> Path:
        prompt = render_prompt(self.config, mode, docs, True)
        result = self.kimi.invoke(prompt, f"sample-{mode}-{self.config.get('prompt_version')}-{int(time.time())}", int(self.config.get("timeout_seconds", 180)), int(self.config.get("max_retries", 2)))
        if result.returncode != 0:
            raise RuntimeError(f"kimi sample generation failed for {mode}: logs={result.stderr_path}")
        rows = parse_jsonl_payload(result.stdout_path.read_text(encoding="utf-8", errors="replace"))
        if not rows:
            raise RuntimeError(f"kimi sample generation produced no JSONL for {mode}: logs={result.stdout_path}")
        write_jsonl_atomic(target, [normalize_record(row, mode, i, str(self.config.get("prompt_version")), "train") for i, row in enumerate(rows[:docs], 1)])
        return target

    def generate_one_shard(self, mode: str, sample: bool) -> Path | None:
        shard_id, split = self.manifest.reserve(mode), None
        split = split_for_shard(shard_id)
        path = self.output_dir / mode / split / f"shard_{shard_id:06d}.jsonl"
        if path.exists() and self.args.resume:
            return path
        prompt, retries = render_prompt(self.config, mode, int(self.config.get("batch_documents", 12)), sample), 0
        for attempt in range(int(self.config.get("max_retries", 2)) + 1):
            retries = attempt
            result = self.kimi.invoke(prompt, f"{mode}-{shard_id:06d}-try{attempt}", int(self.config.get("timeout_seconds", 240)), 0)
            rows = parse_jsonl_payload(result.stdout_path.read_text(encoding="utf-8", errors="replace"))
            rows = [normalize_record(row, mode, i, str(self.config.get("prompt_version")), split) for i, row in enumerate(rows, 1)]
            if not rows:
                if is_transient_result(result):
                    time.sleep(2**attempt); continue
                self.quarantine_payload(mode, shard_id, result, "no_jsonl_records"); continue
            write_jsonl_atomic(path, rows)
            score = score_paths([path], self.tokenizer)
            status = "valid" if score["valid_documents"] == score["documents"] and score["documents"] else "quarantined"
            if status != "valid" and self.config.get("quarantine_bad_shards", True):
                qpath = self.output_dir / "quarantine" / mode / path.name
                qpath.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(path), qpath); path = qpath
            self.manifest.append_success(path, mode, split, shard_id, rows, score, status, retries, result)
            return path
        self.manifest.append_failure(mode, split, shard_id, retries, "generation_failed")
        return None

    def refine_prompts(self, summary: str) -> str:
        if self.args.fake_kimi:
            return ""
        refiner = self.prompt_dir / "prompt_refiner.txt"
        if not refiner.exists():
            return ""
        prompt = refiner.read_text(encoding="utf-8").replace("{{SAMPLE_SUMMARY}}", summary[:8000])
        result = self.kimi.invoke(prompt, f"refine-{int(time.time())}", int(self.config.get("timeout_seconds", 240)), 0)
        if result.returncode != 0:
            return ""
        candidates = extract_prompt_candidates(result.stdout_path.read_text(encoding="utf-8", errors="replace"))
        version, saved = next_prompt_version(self.prompt_dir), False
        for name, body in candidates.items():
            if prompt_candidate_valid(body, name):
                (self.prompt_dir / f"{name}_{version}.txt").write_text(body, encoding="utf-8")
                saved = True
        return version if saved else ""

    def choose_mode(self) -> str:
        mode = str(self.config.get("mode", "mixed"))
        if mode in {"pretrain", "sft"}:
            return mode
        counts = self.manifest.mode_counts()
        counts = {key: counts[key] + self.pending_counts[key] for key in counts}
        total = counts["pretrain"] + counts["sft"]
        if total == 0:
            return "pretrain"
        return "sft" if counts["sft"] / total < float(self.config.get("sft_ratio", 0.1)) else "pretrain"

    def target_reached(self) -> bool:
        return self.manifest.valid_tokens(str(self.config.get("mode", "mixed"))) >= int(self.config.get("target_tokens", 500_000_000))

    def max_calls_reached(self) -> bool:
        max_calls = int(self.config.get("max_calls", 0))
        return bool(max_calls and self.calls_started >= max_calls)

    def stop_requested(self) -> bool:
        return Path(str(self.config.get("stop_file", DEFAULT_RUN_DIR / "STOP"))).exists()

    def prepare_dirs(self) -> None:
        for path in [self.output_dir, self.run_dir, self.logs_dir, self.output_dir / "quarantine"]:
            path.mkdir(parents=True, exist_ok=True)

    def quarantine_payload(self, mode: str, shard_id: int, result, reason: str) -> None:
        target = self.output_dir / "quarantine" / mode / f"shard_{shard_id:06d}-{reason}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.stdout_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    def print_progress(self) -> None:
        rows, tokens = self.manifest.rows(), self.manifest.valid_tokens(str(self.config.get("mode", "mixed")))
        elapsed = max(1e-9, (time.perf_counter() - self.started) / 3600)
        payload = {
            "event": "progress",
            "tokens": tokens,
            "target_tokens": int(self.config.get("target_tokens", 0)),
            "remaining_tokens": max(0, int(self.config.get("target_tokens", 0)) - tokens),
            "tokens_per_hour": round(tokens / elapsed, 2),
            "valid_shards": sum(1 for row in rows if row.get("validation_status") == "valid"),
            "quarantined_shards": sum(1 for row in rows if row.get("validation_status") == "quarantined"),
            "failed_shards": sum(1 for row in rows if row.get("validation_status") == "failed"),
            "retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
        }
        print(json.dumps(payload), flush=True)

    def load_optional_tokenizer(self):
        if self.args.fake_kimi:
            return None
        candidates = [Path(str(self.config["tokenizer_json"]))] if self.config.get("tokenizer_json") else []
        candidates += [Path("data/train/tokenizer/tokenizer.json"), Path("/app/data/tokenizer/tokenizer.json")]
        for path in candidates:
            tokenizer = load_tokenizer(path)
            if tokenizer is not None:
                return tokenizer
        return None
