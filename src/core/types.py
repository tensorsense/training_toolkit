from pydantic import BaseModel, Field, ConfigDict
from typing import Callable, Type


class Preset(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    hf_model_id: str
    hf_model_cls: Type
    hf_processor_cls: Type

    use_qlora: bool = False
    use_lora: bool = False
    find_linear_names_fn: Callable

    batch_size: int = 8

    def as_kwargs(self):
        return self.model_dump()