from pydantic import BaseModel, Field, ConfigDict
from typing import Callable, Type, Optional, List, Dict
from datasets import load_from_disk, Dataset
from transformers import TrainingArguments


class ModelPreset(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    hf_model_id: str
    hf_model_cls: Type
    hf_processor_cls: Type

    use_qlora: bool = False
    use_lora: bool = False
    lora_target_modules: List[str] = Field(default_factory=list)

    training_args: Dict
    preprocessor_cls: Optional[Type] = None
    preprocessor_kwargs: Dict = Field(default_factory=dict)

    def as_kwargs(self):
        return self.model_dump()


DEFAULT_FETCH_CALLBACK = load_from_disk


class DataPreset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Optional[Dataset] = None
    path: str = None
    train_test_split: float = 0.2
    collator_cls: Type = None
    fetch_callback: Callable = DEFAULT_FETCH_CALLBACK

    def with_path(self, path):
        return self.model_copy(update={"path": path}, deep=True)

    def as_kwargs(self, apply_train_test_split=True):
        if self.dataset is None:
            self.dataset = self.fetch_callback(self.path)
        
        if apply_train_test_split:
            self.dataset = self.dataset.train_test_split(test_size=self.train_test_split)
            train_dataset = self.dataset["train"].with_format("torch")
            test_dataset = self.dataset["test"].with_format("torch")
        else:
            train_dataset = self.dataset.with_format("torch")
            test_dataset = None

        
        train_dataset = train_dataset.shuffle(seed=42)

        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "collator_cls": self.collator_cls,
        }
