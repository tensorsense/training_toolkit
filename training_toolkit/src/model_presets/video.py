from training_toolkit import ModelPreset
from training_toolkit.src.common.video_readers import get_video_reader

from transformers import (
    LlavaNextVideoForConditionalGeneration,
    AutoProcessor,
)
from datasets import load_from_disk
from datetime import datetime
import os

from pathlib import Path


class LlavaNextVideoPreprocessor:
    cache_name_template = "llava_next_{dataset_name}_{split}_cache"

    def __init__(self, processor, num_frames=8, max_length=256, cache_location=None) -> None:
        self.processor = processor
        self.num_frames = num_frames
        self.max_length = max_length
        self.cache_location = Path(cache_location).resolve()

        self.num_proc=os.cpu_count()
        self.read_video_fn = get_video_reader()

    def __call__(self, dataset, split):
        if self.cache_location is not None:
            cache_dir = self.cache_location.joinpath(self.cache_name_template.format(dataset_name=dataset.info.dataset_name, split=split))

            if cache_dir.exists():
                dataset = load_from_disk(cache_dir)
                return dataset

        dataset = dataset.map(
            self.collate_fn,
            batched=False,
            fn_kwargs={"processor": self.processor, "max_length": self.max_length},
            num_proc=self.num_proc,
        )

        if self.cache_location is not None:
            dataset.save_to_disk(cache_dir)
        return dataset

    def collate_fn(self, sample, processor, max_length):
        """
        We collate to save everything in tensor format to speed-up dataloading process
        Saving the whole video clip (array) along with caption (string) will slow down iteration
        because unprocessed video clip will take up more memory due to higher resolution
        The processed video on the other hand is always 336x336 in size and fixed frame count per clip
        see: https://discuss.huggingface.co/t/slow-iteration-speed-with-and-without-keep-in-memory-true/33587
        """

        video_clip = self.read_video_fn(
            sample["video_path"],
            self.num_frames,
        )  # change to the video decoder you want

        # Let's use chat template to format the prompt correctly
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["text_prompt"]},
                    {"type": "video"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["target_answer"]},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

        batch = processor(
            text=prompt,
            videos=video_clip,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return batch


llava_next_video_preset = ModelPreset(
    hf_model_id="llava-hf/LLaVa-NeXT-Video-7b-hf",
    hf_model_cls=LlavaNextVideoForConditionalGeneration,
    hf_processor_cls=AutoProcessor,
    use_qlora=True,
    use_lora=False,
    lora_target_modules=[
        "o_proj",
        "up_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "down_proj",
        "q_proj",
    ],
    training_args=dict(
        output_dir=f"llava_next_video_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}",
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,

        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,

        learning_rate=2e-05,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,  # we have the model train and eval with fp16 precision
        fp16_full_eval=True,
        optim="adamw_bnb_8bit",  # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
        # report_to = "wandb", # install wand to use this
        # hub_model_id = REPO_ID,
        # push_to_hub = True, # wel'll push the model to hub after each epoch
        # model that was wrapped for QLORA training with peft will not have arguments listed in its signature
        # so we need to pass lable names explicitly to calculate val loss
        label_names=["labels"],
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=os.cpu_count(),  # let's get more workers since iterating on video datasets might be slower in general
    ),
    preprocessor_cls=LlavaNextVideoPreprocessor,
    preprocessor_kwargs={
        "max_length": 256,
        "cache_location": "dataset_cache/",
    }
)
