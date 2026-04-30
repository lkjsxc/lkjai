from lkjai_train.protocol_decode import token_ids


def test_protocol_stop_markers_use_token_ids():
    assert token_ids(StopTokenizer(), ["</action>", "</missing>", "<eos>"]) == {7, 9}


class StopTokenizer:
    def token_to_id(self, token):
        return {"</action>": 7, "<eos>": 9}.get(token)
