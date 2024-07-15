from training_toolkit import ModelPreset

from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
)
from datetime import datetime
import os


paligemma_image_preset = ModelPreset(
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
    training_args=dict(
        output_dir=f"paligemma_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}",
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,

        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  

        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        # optim="adamw_hf",
        optim="paged_adamw_8bit",
        bf16=True,
        
        # report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=os.cpu_count(),
    ),
)
