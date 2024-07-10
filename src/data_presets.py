from src.core.types import DataPreset
import torch
from enum import Enum
from datasets import load_dataset

# class Task(Enum):
#     VIDEO_QA = "video_qa"
#     IMAGE_QA = "image_qa"


class VideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [
                    feat["input_ids"][0] for feat in features
                ],  # each element is one batch only so we slice [0]
                "attention_mask": [feat["attention_mask"][0] for feat in features],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat(
            [feat["pixel_values_videos"] for feat in features], dim=0
        )

        return padded_inputs


class ImageDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = ["answer " + example["question"] for example in examples]
        labels = [example["multiple_choice_answer"] for example in examples]
        images = [example["image"] for example in examples]

        images = [
            torch.cat([image, image, image], dim=0) if image.shape[0] == 1 else image
            for image in images
        ]

        tokens = self.processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
        )
        return tokens


local_video_preset = DataPreset(
    path="msrvtt_1000.hf",
    train_test_split=0.2,
    collator_cls=VideoDataCollatorWithPadding,
)


def fetch_vqa(*args, **kwargs):
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")
    cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
    dataset = dataset.remove_columns(cols_remove)
    return dataset


vqa2_image_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=ImageDataCollatorWithPadding,
    fetch_callback=fetch_vqa,
)
