import re

SCHEMA = "lkjai-agent-jsonl"
CORPUS = "kimi-sft-60m"
DEFAULT_CONFIG = "configs/corpus/kimi_sft_60m.yaml"
DEFAULT_QUARANTINE = "data/corpus/quarantine/kimi-sft-60m"
DEFAULT_PROMOTED = "data/corpus/generated/kimi-sft-60m"
DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_API_BASE = "https://api.moonshot.ai/v1"
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|[A-Za-z0-9_-]{32,})")
SPLITS = ("train", "val", "holdout")
SPLIT_WEIGHTS = {"train": 90, "val": 5, "holdout": 5}

TOOLS = {
    "agent.finish",
    "agent.think",
    "agent.request_confirmation",
    "resource.search",
    "resource.get",
    "resource.history",
    "resource.create",
    "resource.update_resource",
    "resource.delete",
}
MUTATION_TOOLS = {"resource.create", "resource.update_resource", "resource.delete"}
ROLES = {"system", "user", "assistant", "tool"}
TEMPLATE_FAMILIES = {
    "direct-finish": "direct_finish",
    "read-only-retrieval": "read_only_retrieval",
    "mutation-confirmation": "mutation_confirmation",
    "failure-safety-recovery": "failure_safety_recovery",
}
FAMILY_SKILLS = {
    "direct_finish": "grounding",
    "read_only_retrieval": "resource-retrieval",
    "mutation_confirmation": "mutation-confirmation",
    "failure_safety_recovery": "failure-safety-recovery",
}
