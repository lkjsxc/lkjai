import hashlib


def xml_prompt(request: str, context: str, constraints: str) -> str:
    return "<task>\n" f"<request>{request}</request>\n" f"<context>{context}</context>\n" f"<constraints>{constraints}</constraints>\n" "</task>"


def split_for(row_id: str) -> str:
    bucket = int(hashlib.sha1(row_id.encode("utf-8")).hexdigest(), 16) % 10
    return "holdout" if bucket == 0 else "val" if bucket == 1 else "train"
