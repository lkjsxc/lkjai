import json


def evaluate_fixed_suite(paths, threshold: float = 0.0):
    summary = read_json(paths.training_summary) if paths.training_summary.exists() else {}
    metadata = read_json(paths.dataset_metadata) if paths.dataset_metadata.exists() else {}
    tokenizer = read_json(paths.tokenizer_manifest) if paths.tokenizer_manifest.exists() else {}
    minimum_rows, minimum_unique, minimum_tokens = minimums(metadata)
    cases = [
        case("train-split-exists", paths.train_dataset.exists(), str(paths.train_dataset)),
        case("val-split-exists", paths.val_dataset.exists(), str(paths.val_dataset)),
        case("holdout-split-exists", paths.holdout_dataset.exists(), str(paths.holdout_dataset)),
        case("dataset-metadata-exists", paths.dataset_metadata.exists(), str(paths.dataset_metadata)),
        case("training-summary-exists", paths.training_summary.exists(), str(paths.training_summary)),
        case("tokenizer-manifest-exists", paths.tokenizer_manifest.exists(), str(paths.tokenizer_manifest)),
        case("checkpoint-manifest-exists", paths.checkpoint_manifest.exists(), str(paths.checkpoint_manifest)),
        case("export-manifest-exists", paths.export_manifest.exists(), str(paths.export_manifest)),
        case("checkpoint-has-weights", has_checkpoint_weights(paths.checkpoint_final), "checkpoint weights"),
        case("best-checkpoint-has-weights", has_checkpoint_weights(paths.checkpoint_best), "best checkpoint weights"),
        case("summary-has-loss", has_loss_metrics(summary), "loss metrics"),
        case("summary-has-objective", summary.get("objective") in {"causal_lm_full", "assistant_masked_sft"}, "training objective"),
        case("summary-has-step-accounting", has_step_accounting(summary), "microsteps and optimizer steps"),
        case("dataset-large-enough", int(metadata.get("rows", 0)) >= minimum_rows, f"{minimum_rows} rows"),
        case("dataset-unique-enough", int(metadata.get("unique_rows", 0)) >= minimum_unique, f"{minimum_unique} unique rows"),
        case("tokenizer-train-tokens", int(tokenizer.get("train_tokens", 0)) >= minimum_tokens, f"{minimum_tokens} train tokens"),
    ]
    passed = sum(1 for item in cases if item["passed"])
    report = {"threshold": threshold, "pass_rate": passed / len(cases), "passed": passed, "total": len(cases), "cases": cases}
    paths.runs.mkdir(parents=True, exist_ok=True)
    out = paths.runs / "fixed-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def evaluate_generation_sanity(paths, settings, checkpoint: str = "export"):
    from .behavioral import behavioral_model_source
    from .dataset import parse_assistant_xml
    from .generation import LoadedModel

    model_dir, tokenizer_path = behavioral_model_source(paths, settings, checkpoint)
    model = LoadedModel(model_dir, tokenizer_path=tokenizer_path)
    prompts = [
        {"id": "en-hello", "language": "en", "messages": [{"role": "user", "content": "Say hello in one XML action."}]},
        {"id": "ja-hello", "language": "ja", "messages": [{"role": "user", "content": "日本語で短く挨拶してください。XML actionを1つ返してください。"}]},
    ]
    cases = []
    for prompt in prompts:
        output = model.complete(prompt["messages"], max_tokens=96, temperature=0.0)
        invalid_xml = ""
        try:
            parse_assistant_xml(output)
        except ValueError as error:
            invalid_xml = str(error)
        cases.append(
            {
                "id": prompt["id"],
                "language": prompt["language"],
                "valid_xml": not invalid_xml,
                "invalid_xml": invalid_xml,
                "repetition": repetition_score(output),
                "output": output[:500],
            }
        )
    report = {
        "checkpoint": checkpoint,
        "total": len(cases),
        "valid_xml": sum(1 for case in cases if case["valid_xml"]),
        "repetition_cases": sum(1 for case in cases if case["repetition"] >= 0.5),
        "cases": cases,
    }
    paths.runs.mkdir(parents=True, exist_ok=True)
    out = paths.runs / "generation-sanity.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def repetition_score(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 8:
        return 0.0
    pairs = list(zip(tokens, tokens[1:]))
    return 1.0 - (len(set(pairs)) / max(1, len(pairs)))


def case(case_id: str, passed: bool, detail: str) -> dict:
    return {"id": case_id, "passed": bool(passed), "detail": detail}


def has_checkpoint_weights(checkpoint_dir) -> bool:
    return checkpoint_dir.exists() and (checkpoint_dir / "config.json").exists() and (checkpoint_dir / "model.pt").exists()


def has_loss_metrics(summary: dict) -> bool:
    metrics = summary.get("metrics", {})
    return bool(metrics) and any("loss" in str(key).lower() for key in metrics.keys())


def has_step_accounting(summary: dict) -> bool:
    metrics = summary.get("metrics", {})
    return all(key in metrics for key in ["microsteps", "optimizer_steps", "input_tokens_seen", "loss_tokens_seen"])


def minimums(metadata: dict) -> tuple[int, int, int]:
    if metadata.get("rows", 0) and int(metadata.get("rows", 0)) < 100:
        return 3, 3, 32
    return 60_000, 57_000, 450_000_000


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
