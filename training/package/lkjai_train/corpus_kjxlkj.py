import itertools

from .corpus_shared import split_for, xml_prompt
from .kjxlkj_data import HISTORY_WINDOWS, NOTE_BODIES, PREVIEW_BODIES, RESOURCE_REFS, SEARCH_KINDS, SEARCH_TERMS, UPDATE_BODIES, VISIBILITY_RULES
from .public_data import ANGLES
from .rows import confirm_row, meta, tool_only_row


def kjxlkj_rows(limit: int) -> list[dict]:
    rows = search_rows() + fetch_rows() + history_rows() + preview_rows() + mutation_rows()
    if len(rows) < limit:
        raise RuntimeError(f"kjxlkj rows under target: {len(rows)}")
    return rows[:limit]


def search_rows() -> list[dict]:
    rows = []
    for index, (term, kind) in enumerate(itertools.product(SEARCH_TERMS, SEARCH_KINDS), start=1):
        row_id = f"kj-search-raw-{index:05d}"
        prompt = xml_prompt(f"Search kjxlkj resources for {term}.", "", "Return one valid XML action.")
        rows.append(tool_only_row(prompt, "resource.search", {"query": term, "kind": kind}, ["kjxlkj", "resource.search", "tool_selection", f"kind:{kind}"], meta(row_id, "kjxlkj-api", "search", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    combos = itertools.product(SEARCH_TERMS, SEARCH_KINDS, ANGLES, ["search", "locate"])
    for index, (term, kind, angle, variant) in enumerate(combos, start=1):
        row_id = f"kj-search-{index:05d}"
        prompt = xml_prompt(f"{variant.title()} kjxlkj resources.", f"<query>{term}</query><kind>{kind}</kind><angle>{angle}</angle>", "Return the canonical XML tool call.")
        rows.append(tool_only_row(prompt, "resource.search", {"query": term, "kind": kind}, ["kjxlkj", "resource.search", "tool_selection", f"kind:{kind}"], meta(row_id, "kjxlkj-api", "search", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    return rows


def fetch_rows() -> list[dict]:
    rows = []
    for index, ref in enumerate(RESOURCE_REFS, start=1):
        row_id = f"kj-fetch-raw-{index:05d}"
        prompt = xml_prompt(f"Fetch the {ref} resource.", "", "Return one valid XML action.")
        rows.append(tool_only_row(prompt, "resource.fetch", {"ref": ref}, ["kjxlkj", "resource.fetch", "tool_selection"], meta(row_id, "kjxlkj-api", "fetch", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    for index, (ref, angle, variant) in enumerate(itertools.product(RESOURCE_REFS, ANGLES, ["fetch", "read"]), start=1):
        row_id = f"kj-fetch-{index:05d}"
        prompt = xml_prompt(f"{variant.title()} one resource.", f"<ref>{ref}</ref><angle>{angle}</angle>", "Return the canonical XML tool call.")
        rows.append(tool_only_row(prompt, "resource.fetch", {"ref": ref}, ["kjxlkj", "resource.fetch", "tool_selection"], meta(row_id, "kjxlkj-api", "fetch", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    return rows


def history_rows() -> list[dict]:
    rows = []
    for index, ref in enumerate(RESOURCE_REFS, start=1):
        row_id = f"kj-history-raw-{index:05d}"
        prompt = xml_prompt(f"Show history for the {ref} resource.", "", "Return one valid XML action.")
        rows.append(tool_only_row(prompt, "resource.history", {"ref": ref}, ["kjxlkj", "resource.history", "tool_selection"], meta(row_id, "kjxlkj-api", "history", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    for index, (ref, window, variant) in enumerate(itertools.product(RESOURCE_REFS, HISTORY_WINDOWS, ["Inspect", "Summarize"]), start=1):
        row_id = f"kj-history-{index:05d}"
        prompt = xml_prompt(f"{variant} resource history.", f"<ref>{ref}</ref><window>{window}</window>", "Use the history tool.")
        rows.append(tool_only_row(prompt, "resource.history", {"ref": ref}, ["kjxlkj", "resource.history", "tool_selection"], meta(row_id, "kjxlkj-api", "history", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    return rows


def preview_rows() -> list[dict]:
    rows = []
    for index, body in enumerate(PREVIEW_BODIES, start=1):
        row_id = f"kj-preview-raw-{index:05d}"
        prompt = xml_prompt(f"Preview this markdown: {body}", "", "Return one valid XML action.")
        rows.append(tool_only_row(prompt, "resource.preview_markdown", {"body": body}, ["kjxlkj", "resource.preview_markdown", "tool_selection"], meta(row_id, "kjxlkj-api", "preview", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    for index, (body, angle) in enumerate(itertools.product(PREVIEW_BODIES, ANGLES), start=1):
        row_id = f"kj-preview-{index:05d}"
        prompt = xml_prompt("Preview Markdown safely.", f"<body>{body}</body><angle>{angle}</angle>", "Use the preview tool.")
        rows.append(tool_only_row(prompt, "resource.preview_markdown", {"body": body}, ["kjxlkj", "resource.preview_markdown", "tool_selection"], meta(row_id, "kjxlkj-api", "preview", "synthetic/kjxlkj", split=split_for(row_id), toolset="kjxlkj")))
    return rows


def mutation_rows() -> list[dict]:
    rows = []
    for index, (body, update) in enumerate(zip(NOTE_BODIES, UPDATE_BODIES), start=1):
        ref = RESOURCE_REFS[(index - 1) % len(RESOURCE_REFS)]
        create_id = f"kj-create-raw-{index:05d}"
        create_prompt = xml_prompt(f"Create a kjxlkj note with body {body}", "", "Return one valid XML action.")
        rows.append(confirm_row(create_prompt, "resource.create_note", "resource.create_note", {"body": body, "is_private": False}, "Create a kjxlkj note from the supplied draft after explicit confirmation.", ["confirmation", "kjxlkj", "resource.create_note"], meta(create_id, "kjxlkj-api", "create-note", "synthetic/kjxlkj", split=split_for(create_id), toolset="kjxlkj")))
        update_id = f"kj-update-raw-{index:05d}"
        update_prompt = xml_prompt(f"Update the {ref} resource with body {update}", "", "Return one valid XML action.")
        rows.append(confirm_row(update_prompt, "resource.update_resource", "resource.update_resource", {"ref": ref, "body": update, "is_private": False}, "Update the existing kjxlkj resource after explicit confirmation.", ["confirmation", "kjxlkj", "resource.update_resource"], meta(update_id, "kjxlkj-api", "update-resource", "synthetic/kjxlkj", split=split_for(update_id), toolset="kjxlkj")))
    pairs = list(zip(NOTE_BODIES, UPDATE_BODIES))
    combos = itertools.product(pairs, VISIBILITY_RULES, ANGLES)
    for index, ((body, update), rule, angle) in enumerate(combos, start=1):
        ref = RESOURCE_REFS[(index - 1) % len(RESOURCE_REFS)]
        create_id = f"kj-create-{index:05d}"
        create_prompt = xml_prompt("Create a new note.", f"<draft>{body}</draft><rule>{rule}</rule><angle>{angle}</angle>", "Mutations require confirmation.")
        rows.append(confirm_row(create_prompt, "resource.create_note", "resource.create_note", {"body": body, "is_private": False}, "Create a kjxlkj note from the supplied draft after explicit confirmation.", ["confirmation", "kjxlkj", "resource.create_note"], meta(create_id, "kjxlkj-api", "create-note", "synthetic/kjxlkj", split=split_for(create_id), toolset="kjxlkj")))
        update_id = f"kj-update-{index:05d}"
        update_prompt = xml_prompt("Update an existing resource.", f"<ref>{ref}</ref><body>{update}</body><rule>{rule}</rule><angle>{angle}</angle>", "Mutations require confirmation.")
        rows.append(confirm_row(update_prompt, "resource.update_resource", "resource.update_resource", {"ref": ref, "body": update, "is_private": False}, "Update the existing kjxlkj resource after explicit confirmation.", ["confirmation", "kjxlkj", "resource.update_resource"], meta(update_id, "kjxlkj-api", "update-resource", "synthetic/kjxlkj", split=split_for(update_id), toolset="kjxlkj")))
    return rows
