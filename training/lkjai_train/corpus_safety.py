import itertools

from .corpus_shared import split_for, xml_prompt
from .public_data import SAFER_ALTERNATIVES, SAFETY_BOUNDARIES, SAFETY_REQUESTS
from .rows import direct_row, meta


def safety_rows(limit: int, rules: list[str]) -> list[dict]:
    rows = []
    combos = itertools.product(SAFETY_BOUNDARIES, SAFETY_REQUESTS, rules, SAFER_ALTERNATIVES)
    for index, (boundary, request, rule, safer) in enumerate(combos, start=1):
        row_id = f"safety-{index:05d}"
        prompt = xml_prompt(
            request.format(boundary=boundary),
            f"<boundary>{boundary}</boundary><rule>{rule}</rule>",
            "State the restriction and the safer alternative.",
        )
        answer = f"Do not treat {boundary} as generally writable or publishable state. Preserve the rule that {rule}, then {safer}."
        rows.append(
            direct_row(
                prompt,
                answer,
                ["direct_answer", "language:en", "safety", "visibility_boundary"],
                meta(row_id, "safety-policy", "boundary-answer", "synthetic/safety", split=split_for(row_id), safety_scope="restricted"),
            )
        )
        if len(rows) >= limit:
            return rows
    return rows
