import json
import os
from pathlib import Path

from .corpus_source import validate_sources
from .dataset import prepare_corpus, prepare_fixtures, validate_dataset
from .manifest import export_manifest


def train_pipeline(paths, settings):
    from .behavioral import evaluate_behavior
    from .evals import evaluate_fixed_suite, evaluate_generation_sanity
    from .cli import run_tokenizer, run_training

    validate_sources()
    prepare_fixtures(paths)
    dataset_path = prepare_corpus(paths, settings.corpus_size)
    maybe_prepare_public_pretrain(paths)
    run_tokenizer(paths, settings)
    validate_training_datasets(paths, dataset_path)
    run_training(paths, settings)
    sft_settings = sft_stage_settings(paths, settings)
    run_training(paths, sft_settings)
    export_manifest(paths, sft_settings)
    fixed = evaluate_fixed_suite(paths, sft_settings.fixed_eval_threshold)
    sanity = evaluate_generation_sanity(paths, sft_settings, "export")
    behavioral = evaluate_behavior(paths, sft_settings, sft_settings.behavioral_threshold, "export")
    enforce_acceptance(sft_settings, fixed, sanity, behavioral)
    return behavioral


def train_sft(paths, settings):
    from .cli import run_training

    return run_training(paths, sft_stage_settings(paths, settings))


def sft_stage_settings(paths, settings):
    settings.objective = "assistant_masked_sft"
    if not settings.init_checkpoint:
        settings.init_checkpoint = str(paths.checkpoint_best)
    if "TRAIN_RESUME" not in os.environ:
        settings.resume = "never"
    return settings


def validate_training_datasets(paths, dataset_path: Path) -> None:
    for path in [dataset_path, paths.train_dataset, paths.val_dataset, paths.holdout_dataset]:
        validate_dataset(path)
    if paths.committed_kimi_corpus.exists() and any(paths.committed_kimi_corpus.rglob("*.jsonl")):
        validate_dataset(paths.committed_kimi_corpus)


def maybe_prepare_public_pretrain(paths) -> None:
    from .public_pretrain import prepare_public_pretrain, public_pretrain_sources
    from .public_pretrain_readers import source_paths

    root = Path(os.environ.get("TRAIN_PUBLIC_DATA_DIR", str(paths.root / "raw" / "cosmopedia")))
    sources = public_pretrain_sources()
    if any(source_paths(root, source) for source in sources):
        prepare_public_pretrain(paths)


def enforce_acceptance(settings, fixed: Path, sanity: Path, behavioral: Path) -> None:
    if not fixed_artifact_passes(fixed, settings.fixed_eval_threshold):
        raise RuntimeError(f"fixed artifact gate failed: {failed_case_ids(fixed)}")
    if not generation_sanity_passes(sanity):
        raise RuntimeError("generation sanity gate failed")
    if settings.enforce_competency and not competency_passes(behavioral, settings):
        raise RuntimeError("agent competency gate failed")


def competency_passes(path: Path, settings) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        data.get("xml_validity", 0.0) >= settings.xml_validity_threshold
        and data.get("everyday_chat_pass_rate", 0.0) >= settings.everyday_chat_threshold
        and data.get("pass_rate", 0.0) >= settings.behavioral_threshold
    )


def generation_sanity_passes(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("valid_xml", 0) == data.get("total", 1) and data.get("repetition_cases", 1) == 0


def pass_rate(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8")).get("pass_rate", 0.0))


def fixed_artifact_passes(path: Path, threshold: float) -> bool:
    return pass_rate(path) >= threshold


def failed_case_ids(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    failed = [item["id"] for item in data.get("cases", []) if not item.get("passed")]
    return ", ".join(failed) if failed else "pass_rate below threshold"
