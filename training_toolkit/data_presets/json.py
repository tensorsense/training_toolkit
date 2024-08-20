from training_toolkit import DataPreset
from training_toolkit.common.tokenization_utils.json import JSONTokenizer
from training_toolkit.common.video_readers import get_video_reader
import os


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


class VideoJSONCollator:
    def __init__(self, processor, num_frames=8, max_length=256):
        self.processor = processor

        self.num_frames = num_frames
        self.max_length = max_length

        self.num_proc = os.cpu_count()
        self.read_video_fn = get_video_reader()

    def __call__(self, examples):
        samples = []
        for example in examples:

            video = torch.tensor(
                self.read_video_fn(
                    example["video"],
                    self.num_frames,
                )
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "extract JSON."},
                        {"type": "video"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(example["json"])},
                    ],
                },
            ]

            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )

            sample = self.processor(
                text=prompt,
                videos=video,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            samples.append(sample)

        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [
                    sample["input_ids"][0] for sample in samples
                ],  # each element is one batch only so we slice [0]
                "attention_mask": [sample["attention_mask"][0] for sample in samples],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat(
            [sample["pixel_values_videos"] for sample in samples], dim=0
        )
        return padded_inputs


video_json_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=VideoJSONCollator,
)