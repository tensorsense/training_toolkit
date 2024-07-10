from src.core.types import ModelPreset

import torch
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
)


def llava_next_find_linear_names(model):  # bespoke to the model
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# llava_next_video_preset = ModelPreset(
#     hf_model_id="llava-hf/LLaVa-NeXT-Video-7b-hf",
#     hf_model_cls=LlavaNextVideoForConditionalGeneration,
#     hf_processor_cls=AutoProcessor,
#     use_qlora=True,
#     use_lora=False,
#     lora_target_modules=llava_next_find_linear_names,
#     batch_size=8,
# )


paligemma_preset = ModelPreset(
    hf_model_id="google/paligemma-3b-mix-224",
    hf_model_cls=PaliGemmaForConditionalGeneration,
    hf_processor_cls=AutoProcessor,
    use_qlora=True,
    use_lora=False,
    lora_target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    batch_size=8,
)
