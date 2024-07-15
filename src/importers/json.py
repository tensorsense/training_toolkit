from datasets import Dataset, Image
import os
from pathlib import Path
import json
from collections import defaultdict


class ImageJSONImporter:
    def __init__(self) -> None:
        self.num_proc = os.cpu_count()

    def __call__(self, name, image_path, annotation_path):
        image_path = Path(image_path).resolve()
        annotation_path = Path(annotation_path).resolve()

        with annotation_path.open("r") as f:
            json_dicts = [json.loads(line) for line in f.readlines()]

        dataset_dict = defaultdict(list)

        for json_dict in json_dicts:
            dataset_dict["image"].append(image_path.joinpath(json_dict["image"]))
            dataset_dict["json"].append(json_dict["json"])

        dataset = Dataset.from_dict(dataset_dict).cast_column("image", Image())
        dataset.info.dataset_name = name
        return dataset


class VideoJSONImporter:
    def __init__(self) -> None:
        self.num_proc = os.cpu_count()

    def __call__(self, name, video_path, annotation_path):
        video_path = Path(video_path).resolve()
        annotation_path = Path(annotation_path).resolve()

        with annotation_path.open("r") as f:
            json_dicts = [json.loads(line) for line in f.readlines()]

        dataset_dict = defaultdict(list)

        for json_dict in json_dicts:
            dataset_dict["video"].append(video_path.joinpath(json_dict["video"]))
            dataset_dict["json"].append(json_dict["json"])

        dataset = Dataset.from_dict(dataset_dict)
        dataset.info.dataset_name = name
        return dataset
