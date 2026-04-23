SEARCH_TERMS = [
    "deployment notes",
    "weekly research",
    "public roadmap",
    "editor workflow",
    "verification failures",
    "search ranking",
    "release checklist",
    "privacy policy",
    "compose bootstrap",
    "history retention",
    "markdown preview",
    "resource aliases",
    "live capture notes",
    "mobile navigation",
    "snapshot recovery",
    "api routes",
]

RESOURCE_REFS = [
    "release-notes",
    "weekly-research",
    "ops-runbook",
    "launch-video",
    "timeline-index",
    "private-draft",
    "compose-bootstrap",
    "privacy-policy",
    "resource-history",
    "verification-report",
    "preview-contract",
    "api-contract",
    "mobile-shell",
    "settings-roadmap",
    "ops-checklist",
    "public-home",
]

NOTE_BODIES = [
    "# Research\n\n- Review the latest training run.\n- Compare holdout failures.\n",
    "# Deployment\n\n- Validate compose bootstrap.\n- Keep env loading explicit.\n",
    "# API notes\n\n- Prefer typed resource routes.\n- Preserve history and visibility.\n",
    "# Search quality\n\n- Compare aliases.\n- Keep ranking behavior documented.\n",
    "# Runtime\n\n- Track latency changes.\n- Preserve interactive local use.\n",
    "# Verification\n\n- Run compose checks.\n- Save failing screenshots.\n",
    "# Privacy\n\n- Keep private notes isolated.\n- Require explicit confirmation for writes.\n",
    "# Documentation\n\n- Rewrite the contract first.\n- Keep implementation aligned.\n",
]

UPDATE_BODIES = [
    "# Research\n\n- Review the latest training run.\n- Add raw holdout metrics.\n",
    "# Deployment\n\n- Validate compose bootstrap.\n- Document `.env` before verify.\n",
    "# API notes\n\n- Prefer `/api/resources/...` routes.\n- Preserve history and visibility.\n",
    "# Search quality\n\n- Compare aliases.\n- Add JSON search coverage.\n",
    "# Runtime\n\n- Track latency changes.\n- Add cache-aware inference notes.\n",
    "# Verification\n\n- Run compose checks.\n- Keep failing cases reproducible.\n",
    "# Privacy\n\n- Keep private notes isolated.\n- Add confirmation requirements.\n",
    "# Documentation\n\n- Rewrite the contract first.\n- Link the verification command.\n",
]

PREVIEW_BODIES = [
    "# Heading\n\nParagraph with **bold** text and a list.\n",
    "## Checklist\n\n- search\n- fetch\n- update\n",
    "A note with a [link](https://example.com) and inline `code`.\n",
    "### Release\n\n> Verify the compose stack before the rollout.\n",
    "| Route | Purpose |\n| --- | --- |\n| /api/resources/search | JSON search |\n",
    "Paragraph with `inline code`, a checklist, and a short quote.\n",
    "# Notes\n\n1. Search first.\n2. Fetch the resource.\n3. Ask before writing.\n",
    "Simple preview body about history, visibility, and typed APIs.\n",
]

VISIBILITY_RULES = [
    "private notes must not be summarized into public notes",
    "history lookups must respect current auth",
    "preview should be read-only and safe for admin use",
    "writes must pause for explicit confirmation",
    "resource fetches must preserve current visibility",
    "machine-facing routes must return JSON, not HTML fragments",
    "search results must not leak hidden resources",
    "history access must not bypass resource permissions",
]

SEARCH_KINDS = ["all", "note", "media"]

HISTORY_WINDOWS = ["latest revision", "recent history", "change timeline", "version history"]
