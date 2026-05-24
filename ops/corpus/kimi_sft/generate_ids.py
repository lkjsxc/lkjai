from pathlib import Path

from .constants import SPLIT_WEIGHTS, TEMPLATE_FAMILIES


def split_for_ordinal(ordinal):
    mod = ordinal % 100
    if mod < SPLIT_WEIGHTS["train"]:
        return "train"
    if mod < SPLIT_WEIGHTS["train"] + SPLIT_WEIGHTS["val"]:
        return "val"
    return "holdout"


def family_for_ordinal(config, ordinal):
    families = config.get("sft_domains") or list(TEMPLATE_FAMILIES)
    if not isinstance(families, list) or not families:
        families = list(TEMPLATE_FAMILIES)
    family = str(families[ordinal % len(families)])
    return TEMPLATE_FAMILIES.get(family, family)


def next_shard_id(root, split):
    ids = []
    for path in (Path(root) / split).glob("shard-*.jsonl"):
        if path.stem.startswith("shard-"):
            try:
                ids.append(int(path.stem.removeprefix("shard-")))
            except ValueError:
                pass
    return max(ids, default=0) + 1
