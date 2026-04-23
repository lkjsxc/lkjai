GENERAL_TOPICS = [
    ("debugging", "reproduce, isolate, fix, verify", "premature guesses"),
    ("code review", "find defects before style nits", "missing regressions"),
    ("incident response", "stabilize before optimizing", "unbounded blast radius"),
    ("documentation", "state defaults and verification", "stale behavior claims"),
    ("tool design", "prefer typed interfaces", "ambiguous side effects"),
    ("evaluation", "measure the real serving path", "inflated proxy scores"),
    ("privacy", "preserve visibility boundaries", "public leakage"),
    ("planning", "lock decisions before coding", "hidden implementation drift"),
    ("dataset curation", "deduplicate before splitting", "token inflation"),
    ("tokenization", "train on the same serialization used at runtime", "schema drift"),
    ("serving latency", "cache reusable state", "recomputing the prompt"),
    ("integration contracts", "define canonical API surfaces", "UI-shaped machine calls"),
    ("verification", "test the same path that ships", "mock-only confidence"),
    ("change management", "commit coherent milestones", "giant unreviewable diffs"),
    ("security review", "block unauthorized writes", "silent privilege expansion"),
    ("observability", "capture precise failure signals", "vague summaries"),
]

AUDIENCES = [
    "an LLM agent maintaining a repository",
    "a server-side note assistant",
    "an engineer validating a compose stack",
    "a reviewer checking a risky refactor",
    "a dataset curator preparing a holdout split",
    "a runtime owner debugging slow inference",
    "a product engineer integrating resource APIs",
    "a release reviewer checking acceptance gates",
]

DELIVERABLES = [
    "a concise explanation",
    "a change checklist",
    "a risk summary",
    "an implementation note",
    "a verification note",
    "a migration brief",
    "a rejection rationale",
    "a rollout checklist",
]

CONSTRAINTS = [
    "keep the answer direct",
    "be explicit about uncertainty",
    "name the main failure mode",
    "prioritize verification over optimism",
    "tie the answer to one measurable outcome",
    "state the default before the exception",
    "avoid claiming success without a check",
    "prefer typed interfaces over ad hoc shell steps",
]

ANGLES = [
    "focus on the default path",
    "highlight the likely regression",
    "name the validation command",
    "explain why the contract exists",
    "connect the design to serving quality",
    "state what should not be optimized first",
]

SAFETY_BOUNDARIES = [
    "/etc/passwd",
    "/var/lib/app.db",
    "/home/user/.ssh/id_rsa",
    "a private note",
    "browser cookies",
    "an unconfirmed write request",
    "an admin-only resource",
    "a hidden deployment token",
]

SAFETY_REQUESTS = [
    "Explain how the assistant should handle {boundary}.",
    "State the safe default when a user asks for {boundary}.",
    "Describe the refusal and redirect policy for {boundary}.",
    "Explain why unrestricted access to {boundary} is unsafe.",
]

SAFER_ALTERNATIVES = [
    "ask for explicit confirmation and a typed mutation payload",
    "limit the workflow to authorized resource APIs",
    "return a boundary explanation and stop before the write",
    "suggest a read-only or preview-only alternative",
]

PUBLIC_SOURCE_METADATA = [
    {"name": "lkjai-curated-general", "license": "Apache-2.0"},
]
