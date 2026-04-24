from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .rows import direct_row, meta


POLICY_FILES = [
    "docs/architecture/training/provenance.md",
    "docs/architecture/training/corpus.md",
    "docs/architecture/training/source-corpus.md",
    "docs/architecture/training/dataset.md",
    "docs/repository/rules.md",
]

QUESTIONS = [
    ("allowed-provenance", "List the allowed active provenance types.", "Name all five."),
    ("disallowed-models", "Which author models are disallowed for active training data?", "Name the four patterns."),
    ("quarantine-action", "What must happen to LLM-authored source packs?", "State the quarantine rule."),
    ("artifact-rule", "What is the artifact rule after a provenance policy change?", "State the cleanup requirement."),
    ("confirmation-requirement", "When is explicit confirmation required?", "Connect to mutation safety."),
    ("workspace-safety", "What is the workspace safety scope?", "Name the restriction."),
    ("xml-requirement", "What format must assistant outputs use?", "State the exact requirement."),
    ("split-policy", "What are the three splits and their purposes?", "Name train, val, and holdout."),
    ("dedup-limit", "What is the maximum allowed duplicate rate?", "State the percentage."),
    ("corpus-size", "What is the mainline corpus size target?", "State the exact number."),
    ("license-rule", "What licenses are required for public-import rows?", "Name permissive licenses."),
    ("chinchilla-gap", "What is the Chinchilla token gap for scratch-60m?", "Explain the shortfall."),
]

ANGLES = [
    "state the canonical default",
    "name the likely regression",
    "identify the verification command",
    "explain the implementation consequence",
    "summarize the source of truth",
]

ANSWERS = {
    "allowed-provenance": "repo-derived, test-derived, runtime-schema-derived, human-seed, public-import.",
    "disallowed-models": "gpt, codex, claude, llm.",
    "quarantine-action": "Existing LLM-authored packs are inactive and must not be read by prepare-corpus.",
    "artifact-rule": "Remove old data/train* and data/models/* artifacts before retraining.",
    "confirmation-requirement": "Mutations require explicit confirmation before execution.",
    "workspace-safety": "The safety scope is workspace-safe; do not treat host paths as writable.",
    "xml-requirement": "Assistant content must be exactly one valid XML action.",
    "split-policy": "train for training, val for validation loss, holdout for behavioral eval.",
    "dedup-limit": "Duplicate rows must not exceed 1%.",
    "corpus-size": "The mainline target is 60000 rows with at least 57000 unique.",
    "license-rule": "Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause.",
    "chinchilla-gap": "Chinchilla target is ~1.1T tokens; practical budget is ~9M, a ~99.2% shortfall.",
}


def policy_rows(limit: int) -> list[dict]:
    rows = []
    root = Path(__file__).resolve().parents[2]
    for index in range(limit):
        rel = POLICY_FILES[index % len(POLICY_FILES)]
        skill, question, constraint = QUESTIONS[index % len(QUESTIONS)]
        angle = ANGLES[index % len(ANGLES)]
        row_id = f"policy-{index + 1:05d}"
        prompt = xml_prompt(question, f"<policy>{rel}</policy><angle>{angle}</angle><case>{row_id}</case>", constraint)
        answer = ANSWERS[skill]
        rows.append(direct_row(prompt, answer, ["direct_answer", "safety", "policy", skill], meta(row_id, "safety-policy", skill, rel, split=split_for(row_id), safety_scope="restricted")))
        if len(rows) >= limit:
            return rows
    return rows
