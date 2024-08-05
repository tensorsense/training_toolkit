import re


def json2token(obj, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                output += (
                    rf"<s_{k}>"
                    + json2token(
                        obj[k], sort_json_key  # update_special_tokens_for_json_key,
                    )
                    + rf"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [
                json2token(item, sort_json_key)  # update_special_tokens_for_json_key,
                for item in obj
            ]
        )
    else:
        obj = str(obj)
        return obj


def token2json(processor, tokens, is_inner_value=False, added_vocab=None):
    """
    Convert a (generated) token sequence into an ordered JSON format.
    """
    if added_vocab is None:
        added_vocab = processor.tokenizer.get_added_vocab()

    output = {}

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        key_escaped = re.escape(key)

        end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}",
                tokens,
                re.IGNORECASE | re.DOTALL,
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(
                        content, is_inner_value=True, added_vocab=added_vocab
                    )
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(
                    tokens[6:], is_inner_value=True, added_vocab=added_vocab
                )

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


class JSONTokenizer:
    def __init__(self, processor) -> None:
        self.processor = processor

    def encode(self, obj):
        return json2token(obj, sort_json_key=True)

    def decode(self, tokens):
        return token2json(
            self.processor, tokens, is_inner_value=False, added_vocab=None
        )
