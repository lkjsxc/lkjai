import pytest

from lkjai_train.formatting import prompt_text
from lkjai_train.tokenizer_tokens import XML_TAG_TOKENS


def test_prompt_text_uses_attribute_free_message_tags():
    text = prompt_text([{"role": "user", "content": "hello"}])
    assert "<message role=" not in text
    assert "<message>" in text
    assert "<role>user</role>" in text


def test_xml_like_tags_are_single_token():
    pytest.importorskip("tokenizers")
    from lkjai_train.tokenizer import train_text_tokenizer

    tokenizer = train_text_tokenizer(["hello " + " ".join(XML_TAG_TOKENS)], 512)
    assert tokenizer.get_vocab_size() <= 512
    for tag in XML_TAG_TOKENS:
        encoded = tokenizer.encode(tag).ids
        assert len(encoded) == 1, tag
        assert tokenizer.decode(encoded, skip_special_tokens=True) == tag
