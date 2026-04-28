from xml.sax.saxutils import escape


def fallback_action(text: str) -> str:
    content = text.strip() or "I could not produce a complete model action."
    return f"<action><tool>agent.finish</tool><content>{escape(content)}</content></action>"
