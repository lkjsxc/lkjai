import os
from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .docs_data import doc_records
from .rows import direct_row, meta


def doc_rows(limit: int) -> list[dict]:
    root = Path(os.environ.get("REPO_DOCS_DIR", "/workspace/docs"))
    if not root.exists():
        root = Path(__file__).resolve().parents[2] / "docs"
    prompts = [
        ("summarize-contract", "Summarize the contract defined in {path}.", "Keep the answer grounded in the supplied file."),
        ("state-default", "State the default behavior defined in {path}.", "State the default before exceptions."),
        ("verification-focus", "Explain what verification protects in {path}.", "Name the concrete verification focus."),
        ("migration-note", "Write a migration note for the contract in {path}.", "Mention what older behavior must change."),
        ("failure-mode", "Describe the main failure mode prevented by {path}.", "Connect the answer to one implementation risk."),
        ("llm-reader", "Explain {path} for an LLM agent editing the repository.", "Keep the answer short and machine-readable."),
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
