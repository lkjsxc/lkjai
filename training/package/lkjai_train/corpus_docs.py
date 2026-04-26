import os
from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .docs_data import doc_records
from .rows import direct_row, meta


AUDIENCES = [
    "an LLM agent editing the repository",
    "a reviewer checking implementation drift",
    "an operator running docker compose",
    "a maintainer updating the training path",
    "a contributor adding a new feature",
]

ANGLES = [
    "state the canonical default",
    "name the likely regression",
    "identify the verification command",
    "explain the implementation consequence",
    "summarize the source of truth",
    "map the change to an existing decision",
]


def doc_rows(limit: int) -> list[dict]:
    root = Path(os.environ.get("REPO_DOCS_DIR", "/workspace/docs"))
    if not root.exists():
        root = Path(__file__).resolve().parents[3] / "docs"
    rows = file_doc_rows(root, limit)
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
        ("active-owner", "Which component owns the contract in {path}?", "Name the owner and one boundary."),
        ("data-risk", "What data risk does {path} prevent?", "Tie the risk to training or runtime behavior."),
        ("acceptance", "What acceptance signal appears in {path}?", "Name a measurable signal."),
        ("decision-link", "Which decision does {path} implement?", "Name the decision file and rationale."),
        ("compose-check", "How does {path} affect docker compose behavior?", "Name one profile or service."),
        ("token-impact", "How does {path} affect tokenizer or model training?", "Name one concrete parameter or budget."),
    ]
    rows = []
    for index, record in enumerate(doc_records(root), start=1):
        for skill, request, constraint in prompts:
            for audience in AUDIENCES:
                for angle in ANGLES:
                    row_id = f"docs-{index:04d}-{len(rows) + 1:05d}"
                    context = f"<path>{record['path']}</path><title>{record['title']}</title><audience>{audience}</audience><angle>{angle}</angle>"
                    prompt = xml_prompt(request.format(path=record["path"]), context, f"{constraint} {angle}.")
                    answer = doc_answer(record, audience, angle)
                    rows.append(direct_row(prompt, answer, ["direct_answer", "docs_grounding", "language:en", skill], meta(row_id, "lkjai-docs", skill, record["path"], split=split_for(row_id))))
                    if len(rows) >= limit:
                        return rows
    return rows


def doc_answer(record: dict, audience: str, angle: str) -> str:
    parts = [record["title"] + ":", record["snippet"], record["defaults"], record["verification"]]
    grounded = " ".join(part for part in parts if part)[:480]
    return f"For {audience}, {angle}: {grounded}"
