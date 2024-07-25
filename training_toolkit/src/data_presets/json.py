from training_toolkit import DataPreset

import torch
import json


PROMPT = "extract JSON."


class ImageJSONCollatorWithPadding:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        json_dicts = [json.loads(example["json"]) for example in examples]
        labels = [self.json2token(json_dict) for json_dict in json_dicts]

        images = [example["image"] for example in examples]

        images = [
            self.fix_image_channels(image) if image.shape[0] != 3 else image
            for image in images
        ]

        texts = [PROMPT for _ in range(len(examples))]

        try:
            tokens = self.processor(
                text=texts,
                images=images,
                suffix=labels,
                return_tensors="pt",
                padding="longest",
            )
        except Exception as e:
            for image in images:
                print(image.shape)
            raise e
        return tokens

    @staticmethod
    def fix_image_channels(image):
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)
        elif image.shape[0] > 3:
            image = image[:3]
        return image

    def json2token(
        self,
        obj,
        # update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
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
                        + self.json2token(
                            obj[k], sort_json_key  # update_special_tokens_for_json_key,
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [
                    self.json2token(
                        item, sort_json_key  # update_special_tokens_for_json_key,
                    )
                    for item in obj
                ]
            )
        else:
            obj = str(obj)
            return obj


image_json_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=ImageJSONCollatorWithPadding,
)
