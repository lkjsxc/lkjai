from dataclasses import dataclass

from .formatting import row_text, supervised_token_ids


CAUSAL_LM_FULL = "causal_lm_full"
ASSISTANT_MASKED_SFT = "assistant_masked_sft"
OBJECTIVES = {CAUSAL_LM_FULL, ASSISTANT_MASKED_SFT}


@dataclass(frozen=True)
class ObjectiveTokens:
    ids: list[int]
    loss_mask: list[int]


def normalize_objective(objective: str) -> str:
    aliases = {
        "scratch-causal-lm": CAUSAL_LM_FULL,
        "causal_lm": CAUSAL_LM_FULL,
        "full": CAUSAL_LM_FULL,
        "sft": ASSISTANT_MASKED_SFT,
        "assistant": ASSISTANT_MASKED_SFT,
        "supervised": ASSISTANT_MASKED_SFT,
    }
    value = aliases.get(objective, objective)
    if value not in OBJECTIVES:
        raise ValueError(f"unknown TRAIN_OBJECTIVE={objective}")
    return value


def objective_tokens(tokenizer, row: dict, objective: str) -> ObjectiveTokens:
    objective = normalize_objective(objective)
    if objective == CAUSAL_LM_FULL:
        ids = tokenizer.encode(row_text(row)).ids
        return ObjectiveTokens(ids, [1] * len(ids))
    ids, labels = supervised_token_ids(tokenizer, row)
    return ObjectiveTokens(ids, [0 if label == -100 else 1 for label in labels])

