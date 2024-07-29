from training_toolkit import DataPreset
from training_toolkit.src.common.tokenization_utils.json import JSONTokenizer

import torch
import json


JSON_PROMPT = "extract JSON."


class ImageJSONCollatorWithPadding:

    def __init__(self, processor):
        self.processor = processor
        self.json_tokenizer = JSONTokenizer(processor)

    def __call__(self, examples):
        json_dicts = [json.loads(example["json"]) for example in examples]
        labels = [self.json_tokenizer.encode(json_dict) for json_dict in json_dicts]

        images = [example["image"] for example in examples]

        images = [
            self.fix_image_channels(image) if image.shape[0] != 3 else image
            for image in images
        ]

        texts = [JSON_PROMPT for _ in range(len(examples))]

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


image_json_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=ImageJSONCollatorWithPadding,
)
