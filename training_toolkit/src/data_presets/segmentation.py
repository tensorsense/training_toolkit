from training_toolkit import DataPreset
from training_toolkit.src.common.tokenization_utils.segmentation import (
    SegmentationTokenizer,
)

import torch


class ImageSegmentationCollator:

    def __init__(self, processor):
        self.processor = processor
        self.segmentation_tokenizer = SegmentationTokenizer()

    def __call__(self, examples):

        # image, image_path, prefix, xyxy_list, mask_list, class_id_list, classes
        prefix = "segment " + " ; ".join(ds.classes)
        images = [example["image"] for example in examples]

        images = [
            self.fix_image_channels(image) if image.shape[0] != 3 else image
            for image in images
        ]

        texts = [prefix for _ in range(len(examples))]

        labels = self.segmentation_tokenizer(
            image, image_path, xyxy_list, mask_list, class_id_list, classes
        )

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


image_segmentation_preset = DataPreset(
    train_test_split=0.2,
    collator_cls=ImageSegmentationCollator,
)
