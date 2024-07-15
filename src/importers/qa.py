from datasets import Dataset, Image
import pandas as pd
import os
from pathlib import Path
import json
from typing import Generator, Dict


def load_qa(qa_path: Path, vision_key="video") -> pd.DataFrame:
    # read annotations from disk

    with qa_path.open("r") as f:
        qa = json.load(f)

    samples = []
    for entry in qa:
        for i, conversation in enumerate(entry["conversations"]):
            assert conversation[0]["from"] == "human"
            assert conversation[1]["from"] == "gpt"

            vision = entry.get(vision_key, None)
            assert vision is not None

            samples.append(
                {
                    vision_key: vision,
                    "messages": conversation,
                    "question_id": f"{entry['video'][:-4]}_{i}",
                }
            )

    df = pd.DataFrame(samples)
    return df


def qa_generator(
    vision_key,
) -> Generator[Dict, None, None]:
    def _generate(vision_path: Path, qa_path: Path):
        df = load_qa(qa_path, vision_key)

        for idx, row in df.iterrows():
            sample = {
                f"{vision_key}_path": Path(vision_path)
                .joinpath(row.video)
                .resolve()
                .as_posix(),
                "text_prompt": row.messages[0]["value"],
                "target_answer": row.messages[1]["value"],
                "question_id": row.question_id,
            }

            yield sample
            if idx >= 1000:
                break

    return _generate


class ImageQAImporter:
    def __init__(self) -> None:
        self.generator = qa_generator("image")
        self.num_proc = os.cpu_count()

    def __call__(self, name, image_path, annotation_path):
        dataset = Dataset.from_generator(
            self.generator,
            gen_kwargs={
                "vision_path": Path(image_path),
                "qa_path": Path(annotation_path),
            },
        )
        dataset = dataset.cast_column("image", Image())
        dataset.info.dataset_name = name
        return dataset


class VideoQAImporter:
    def __init__(self) -> None:
        self.generator = qa_generator("video")
        self.num_proc = os.cpu_count()

    def __call__(self, name, video_path, annotation_path):
        dataset = Dataset.from_generator(
            self.generator,
            gen_kwargs={
                "vision_path": Path(video_path),
                "qa_path": Path(annotation_path),
            },
        )

        dataset.info.dataset_name = name
        return dataset
