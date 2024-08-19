from training_toolkit import DataPreset
from training_toolkit.common.video_readers import get_video_reader
import torch
import os


class VideoQACollatorWithPadding:
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
                    example["video_path"],
                    self.num_frames,
                )
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example["text_prompt"]},
                        {"type": "video"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["target_answer"]},
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


class ImageQACollatorWithPadding:
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


video_qa_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=VideoQACollatorWithPadding,
)

image_qa_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=ImageQACollatorWithPadding,
)
