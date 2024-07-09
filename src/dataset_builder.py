from transformers import AutoProcessor
from datasets import concatenate_datasets, Dataset, load_from_disk
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import torch

from pathlib import Path
import json
from typing import Generator, Dict

from decord import VideoReader, gpu, cpu


def read_video_decord(video_path, num_frames):
    """
    Decode the video with Decord decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    vr = VideoReader(
        uri=video_path, ctx=cpu(0)
    )  # you need to install from source to use gpu ctx
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def load_qa(qa_path: Path) -> pd.DataFrame:
    # read annotations from disk

    with qa_path.open("r") as f:
        qa = json.load(f)

    samples = []
    for entry in qa:
        for i, conversation in enumerate(entry["conversations"]):
            assert conversation[0]["from"] == "human"
            assert conversation[1]["from"] == "gpt"

            samples.append(
                {
                    "video": entry["video"],
                    "messages": conversation,
                    "question_id": f"{entry['video'][:-4]}_{i}",
                }
            )

    df = pd.DataFrame(samples)
    return df


def prepare_batches(video_path: Path, qa_path: Path) -> Generator[Dict, None, None]:
    df = load_qa(qa_path)

    for idx, row in df.iterrows():
        sample = {
            "video_path": Path(video_path).joinpath(row.video).resolve().as_posix(),
            "text_prompt": row.messages[0]["value"],
            "target_answer": row.messages[1]["value"],
            "question_id": row.question_id,
        }

        yield sample
        if idx >= 1000:
            break


# We collate to save everything in tensor format to speed-up dataloading process
# Saving the whole video clip (array) along with caption (string) will slow down iteration
# because unprocessed video clip will take up more memory due to higher resolution
# The processed video on the other hand is always 336x336 in size and fixed frame count per clip
# see: https://discuss.huggingface.co/t/slow-iteration-speed-with-and-without-keep-in-memory-true/33587


def collate_fn(sample, processor, max_length):
    video_clip = read_video_decord(
        sample["video_path"]
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


def build_dataset(
    video_path, qa_path, dataset_name, hf_processor_cls, hf_model_id, num_frames, max_length
):
    # And we also need to load the processor for collate_fn
    processor = hf_processor_cls.from_pretrained(hf_model_id, use_fast=False)
    processor.tokenizer.padding_side = (
        "right"  # during training, one always uses padding on the right
    )

    ds = Dataset.from_generator(
        prepare_batches,
        gen_kwargs={
            "video_path": video_path,
            "qa_path": qa_path,
            "num_frames": num_frames,
        },
    )

    dataset = ds.map(
        collate_fn,
        batched=False,
        fn_kwargs={"processor": processor, "max_length": max_length},
        num_proc=8,
    )

    dataset.save_to_disk(dataset_name)
    return dataset


def fetch_local_dataset(dataset_name):

    dataset = load_from_disk(dataset_name)

    # dataset = dataset.shuffle(seed=42)
    # dataset = dataset.train_test_split(test_size=0.2)

    return {
        "train_dataset": dataset["train"].with_format("torch"),
        "test_dataset": dataset["test"].with_format("torch"),
    }
