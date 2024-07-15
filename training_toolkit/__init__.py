from src.train_builder import build_trainer

from src.importers.qa import ImageQAImporter, VideoQAImporter
from src.importers.json import ImageJSONImporter, VideoJSONImporter

from src.core.types import DataPreset, ModelPreset

from src.model_presets import paligemma_preset, llava_next_video_preset
from src.data_presets import vqa2_image_preset, local_video_preset