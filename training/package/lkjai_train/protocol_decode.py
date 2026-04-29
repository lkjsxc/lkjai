from html import escape

from .formatting import prompt_text


FINISH_PREFIX = "<action>\n<reasoning>Answer the user directly.</reasoning>\n<tool>agent.finish</tool>\n<content>"
FINISH_SUFFIX = "</content>\n</action>"


def raw_completion(loaded, messages: list[dict], max_tokens: int, temperature: float) -> str:
    import torch

    prompt_ids = loaded.tokenizer.encode(prompt_text(messages)).ids[-loaded.config.sequence_len :]
    return decode_after_ids(loaded, torch.tensor([prompt_ids], device=loaded.device), max_tokens, temperature, stop_markers=["</action>"])


def constrained_finish(loaded, messages: list[dict], max_tokens: int, temperature: float) -> str:
    import torch

    prompt = prompt_text(messages) + FINISH_PREFIX
    prompt_ids = loaded.tokenizer.encode(prompt).ids[-loaded.config.sequence_len :]
    generated = decode_after_ids(
        loaded,
        torch.tensor([prompt_ids], device=loaded.device),
        max_tokens,
        temperature,
        stop_markers=["</content>", "</action>", "<eos>"],
    )
    return FINISH_PREFIX + clean_content(generated) + FINISH_SUFFIX


def decode_after_ids(loaded, input_ids, max_tokens: int, temperature: float, stop_markers: list[str]) -> str:
    import torch

    generated = []
    eos = loaded.tokenizer.token_to_id("<eos>")
    with torch.inference_mode():
        logits, _, cache = loaded.model(input_ids, use_cache=True)
        next_logits = logits[:, -1, :]
        for _ in range(max_tokens):
            next_id = loaded.choose_token(next_logits, temperature)
            token = int(next_id.item())
            generated.append(token)
            text = loaded.tokenizer.decode(generated, skip_special_tokens=False)
            if eos is not None and token == eos:
                break
            if any(marker in text for marker in stop_markers):
                break
            logits, _, cache = loaded.model(next_id, cache=cache, use_cache=True)
            next_logits = logits[:, -1, :]
    return loaded.tokenizer.decode(generated, skip_special_tokens=False)


def clean_content(text: str) -> str:
    text = before_any(text, ["</content>", "</action>", "<eos>"])
    for special in ["<pad>", "<unk>", "<bos>", "<assistant_action>"]:
        text = text.replace(special, "")
    text = text.replace("&lt;", "<").replace("&amp;", "&").strip()
    text = strip_action_fragments(text).strip()
    if not text:
        text = "I could not produce a useful answer yet."
    return escape(text, quote=False)


def before_any(text: str, markers: list[str]) -> str:
    stops = [index for marker in markers if (index := text.find(marker)) >= 0]
    return text[: min(stops)] if stops else text


def strip_action_fragments(text: str) -> str:
    for marker in ["<action>", "<reasoning>", "</reasoning>", "<tool>", "</tool>", "<content>"]:
        text = text.replace(marker, "")
    return text
