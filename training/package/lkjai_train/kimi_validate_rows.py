from dataclasses import dataclass

from .dataset import parse_assistant_xml
from .formatting import row_text


SFT_META = {
    "id",
    "split",
    "provenance",
    "author_type",
    "author_model",
    "quality_tier",
    "domain",
    "skill",
    "toolset",
    "language",
    "safety_scope",
    "license",
    "source_ref",
    "mode",
    "prompt_version",
}


PRETRAIN_META = {
    "source",
    "mode",
    "generated_at",
    "prompt_version",
    "estimated_tokens",
    "provenance",
    "author_type",
    "author_model",
    "language",
    "license",
    "source_ref",
}


@dataclass(frozen=True)
class KimiRowFacts:
    mode: str
    split: str
    language: str
    domain: str
    license: str
    xml_total: int
    xml_valid: int
    final_tool: str
    flags: list[str]

    @property
    def valid(self) -> bool:
        return not self.flags


def validate_kimi_row(row: dict) -> KimiRowFacts:
    mode = row_mode(row)
    if mode == "pretrain":
        return validate_pretrain(row)
    if mode == "sft":
        return validate_sft(row)
    return KimiRowFacts(mode, "unknown", "unknown", "unknown", "unknown", 0, 0, "", ["unknown_mode"])


def row_mode(row: dict) -> str:
    if row.get("mode") == "pretrain" or "text" in row:
        return "pretrain"
    meta = row.get("meta", {})
    if meta.get("mode") == "sft" or "messages" in row:
        return "sft"
    return "unknown"


def validate_pretrain(row: dict) -> KimiRowFacts:
    meta = row.get("metadata", {})
    flags = []
    required = {"id", "mode", "language", "domain", "difficulty", "title", "text", "metadata"}
    if required - row.keys():
        flags.append("missing_pretrain_fields")
    if not isinstance(meta, dict) or PRETRAIN_META - meta.keys():
        flags.append("missing_pretrain_meta_fields")
    if row.get("mode") != "pretrain" or meta.get("mode") != "pretrain":
        flags.append("wrong_pretrain_mode")
    if row.get("language") != "en" or meta.get("language") != "en":
        flags.append("non_english_row")
    if meta.get("source") != "kimi_synthetic":
        flags.append("bad_pretrain_source")
    if meta.get("provenance") != "kimi-generated":
        flags.append("bad_pretrain_provenance")
    if meta.get("author_type") != "external-agent-generated":
        flags.append("bad_pretrain_author_type")
    if meta.get("author_model") != "kimi-code":
        flags.append("bad_pretrain_author_model")
    if meta.get("license") != "project-local":
        flags.append("bad_pretrain_license")
    if not isinstance(row.get("text"), str) or len(row.get("text", "").strip()) < 80:
        flags.append("short_or_empty_text")
    if not english_text(row_text(row)):
        flags.append("language_mismatch")
    return KimiRowFacts(
        "pretrain",
        str(meta.get("split", "unknown")),
        str(row.get("language", meta.get("language", "unknown"))),
        str(row.get("domain", "unknown")),
        str(meta.get("license", "unknown")),
        0,
        0,
        "",
        sorted(set(flags)),
    )


def validate_sft(row: dict) -> KimiRowFacts:
    meta = row.get("meta", {})
    messages = row.get("messages", [])
    tags = set(row.get("tags", []))
    flags = []
    if not isinstance(messages, list) or len(messages) < 2:
        flags.append("bad_messages")
        messages = []
    if not isinstance(meta, dict) or SFT_META - meta.keys():
        flags.append("missing_sft_meta_fields")
    if meta.get("provenance") != "kimi-generated":
        flags.append("bad_sft_provenance")
    if meta.get("author_type") != "external-agent-generated" or meta.get("author_model") != "kimi-code":
        flags.append("bad_sft_author")
    if meta.get("mode") != "sft":
        flags.append("wrong_sft_mode")
    if meta.get("language") != "en":
        flags.append("non_english_row")
    if meta.get("license") != "project-local":
        flags.append("bad_sft_license")
    if "preference" in tags or meta.get("domain") == "preferences":
        flags.append("preference_row_in_active_corpus")
    xml_total, xml_valid, final_tool = validate_assistant_messages(messages, flags)
    if final_tool != "agent.finish":
        flags.append("missing_final_agent_finish")
    if not english_text(row_text(row)):
        flags.append("language_mismatch")
    return KimiRowFacts(
        "sft",
        str(meta.get("split", "unknown")),
        str(meta.get("language", "unknown")),
        str(meta.get("domain", "unknown")),
        str(meta.get("license", "unknown")),
        xml_total,
        xml_valid,
        final_tool,
        sorted(set(flags)),
    )


def validate_assistant_messages(messages: list[dict], flags: list[str]) -> tuple[int, int, str]:
    total = valid = 0
    final_tool = ""
    for message in messages:
        role = message.get("role")
        if role != "assistant":
            continue
        total += 1
        try:
            parsed = parse_assistant_xml(str(message.get("content", "")))
        except ValueError:
            flags.append("invalid_assistant_xml")
            continue
        valid += 1
        final_tool = str(parsed.get("tool", ""))
    if total == 0:
        flags.append("missing_assistant")
    return total, valid, final_tool


def english_text(text: str) -> bool:
    if not text.strip():
        return False
    non_ascii_letters = sum(1 for ch in text if ord(ch) > 127 and ch.isalpha())
    return non_ascii_letters / max(1, len(text)) < 0.03
