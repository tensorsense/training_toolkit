from datasets import Dataset
import pandas as pd
import os
from pathlib import Path
import json
from typing import Generator, Dict


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


class VideoImporter:
    def __init__(self) -> None:
        self.annotations_generator = prepare_batches
        self.num_proc = os.cpu_count()

    def __call__(self, name, video_path, annotation_path):
        dataset = Dataset.from_generator(
            self.annotations_generator,
            gen_kwargs={
                "video_path": Path(video_path),
                "qa_path": Path(annotation_path),
            },
        )

        dataset.info.dataset_name = name
        return dataset
