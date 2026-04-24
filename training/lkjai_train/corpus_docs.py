import itertools
import os
from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .corpus_source import tagged_contents
from .docs_data import doc_records
from .public_data import ANGLES, CONSTRAINTS
from .rows import direct_row, meta


def doc_rows(limit: int) -> list[dict]:
    root = Path(os.environ.get("REPO_DOCS_DIR", "/workspace/docs"))
    if not root.exists():
        root = Path(__file__).resolve().parents[2] / "docs"
    rows = file_doc_rows(root, limit)
    if len(rows) < limit:
        rows += grounding_rows(limit - len(rows))
    if len(rows) < limit:
        raise RuntimeError(f"doc rows under target: {len(rows)}")
    return rows[:limit]


def file_doc_rows(root: Path, limit: int) -> list[dict]:
    prompts = [
        ("summarize-contract", "Summarize the contract defined in {path}.", "Keep the answer grounded in the supplied file."),
        ("state-default", "State the default behavior defined in {path}.", "State the default before exceptions."),
        ("verification-focus", "Explain what verification protects in {path}.", "Name the concrete verification focus."),
        ("migration-note", "Write a migration note for the contract in {path}.", "Mention what older behavior must change."),
        ("failure-mode", "Describe the main failure mode prevented by {path}.", "Connect the answer to one implementation risk."),
        ("llm-reader", "Explain {path} for an LLM agent editing the repository.", "Keep the answer short and machine-readable."),
        ("compare-alternative", "Compare the approach in {path} to an alternative.", "Name one concrete alternative."),
        ("extract-invariant", "Extract an invariant from {path}.", "State the invariant as a single sentence."),
        ("identify-dependency", "What does {path} depend on?", "List at least one upstream dependency."),
        ("state-stop-rule", "What is the stop rule in {path}?", "State the exact condition that halts the process."),
        ("audience-focus", "Who is the intended audience of {path}?", "Name the primary reader."),
        ("update-checklist", "What must be updated when {path} changes?", "List one downstream artifact."),
    ]
    rows = []
    for index, record in enumerate(doc_records(root), start=1):
        for variant, (skill, request, constraint) in enumerate(prompts, start=1):
            row_id = f"docs-{index:04d}-{variant}"
            prompt = xml_prompt(request.format(path=record["path"]), f"<path>{record['path']}</path><title>{record['title']}</title>", constraint)
            answer = " ".join(filter(None, [record["title"] + ":", record["snippet"], record["defaults"], record["verification"]]))[:560]
            rows.append(
                direct_row(
                    prompt,
                    answer,
                    ["direct_answer", "docs_grounding", "language:en", skill],
                    meta(row_id, "lkjai-docs", skill, record["path"], split=split_for(row_id)),
                )
            )
            if len(rows) >= limit:
                return rows
    return rows


def grounding_rows(limit: int) -> list[dict]:
    scenarios = tagged_contents("docs_grounding", "docs_scenario")
    rows = []
    combos = itertools.product(scenarios, ANGLES, CONSTRAINTS)
    for index, (scenario, angle, constraint) in enumerate(combos, start=1):
        row_id = f"docs-ground-{index:05d}"
        prompt = xml_prompt(
            scenario["question"],
            f"<topic>{scenario['doc_topic']}</topic><angle>{angle}</angle>",
            f"{constraint}; {angle}.",
        )
        rows.append(
            direct_row(
                prompt,
                scenario["grounded_answer"],
                ["direct_answer", "docs_grounding", "language:en", "grounded"],
                meta(row_id, "lkjai-docs", "grounding", scenario["source_path_pattern"], split=split_for(row_id)),
            )
        )
        if len(rows) >= limit:
            return rows
    return rows
