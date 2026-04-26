def sample_section(version: str, summary: dict) -> str:
    keys = [
        "documents",
        "valid_documents",
        "approx_tokens",
        "duplicate_rate",
        "near_duplicate_rate",
        "mean_score",
        "flag_counts",
    ]
    lines = [f"## Prompt {version}", ""]
    lines += [f"- {key}: `{summary.get(key, 0)}`" for key in keys]
    return "\n".join(lines + [""])
