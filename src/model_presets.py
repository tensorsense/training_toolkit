from src.core.types import ModelPreset

from transformers import (
    LlavaNextVideoForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)

from datetime import datetime
import os


llava_next_video_preset = ModelPreset(
    hf_model_id="llava-hf/LLaVa-NeXT-Video-7b-hf",
    hf_model_cls=LlavaNextVideoForConditionalGeneration,
    hf_processor_cls=AutoProcessor,
    use_qlora=True,
    use_lora=False,
    lora_target_modules=[
        "o_proj",
        "up_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "down_proj",
        "q_proj",
    ],
    training_args=dict(
        output_dir=f"llava_next_video_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}",
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,

        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,

        learning_rate=2e-05,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,  # we have the model train and eval with fp16 precision
        fp16_full_eval=True,
        optim="adamw_bnb_8bit",  # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
        # report_to = "wandb", # install wand to use this
        # hub_model_id = REPO_ID,
        # push_to_hub = True, # wel'll push the model to hub after each epoch
        # model that was wrapped for QLORA training with peft will not have arguments listed in its signature
        # so we need to pass lable names explicitly to calculate val loss
        label_names=["labels"],
        dataloader_num_workers=os.cpu_count(),  # let's get more workers since iterating on video datasets might be slower in general
    ),
)


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
