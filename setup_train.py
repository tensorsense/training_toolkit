import numpy as np

from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextVideoForConditionalGeneration,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch


class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [
                    feat["input_ids"][0] for feat in features
                ],  # each element is one batch only so we slice [0]
                "attention_mask": [feat["attention_mask"][0] for feat in features],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat(
            [feat["pixel_values_videos"] for feat in features], dim=0
        )

        return padded_inputs


def find_all_linear_names(model):  # bespoke to the model
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


def setup_train(use_qlora, use_lora, model_id, ouput_dir, batch_size, processor, train_dataset, test_dataset):
    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
    # Standard LoRA:  model is loaded with standard LoRA adaptations.
    # Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

    if use_qlora or use_lora:
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto",
        )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)


    args = TrainingArguments(
        # args related to training
        output_dir=ouput_dir,
        eval_strategy="steps",
        eval_steps=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-05,
        max_steps=100,  # adjust this depending on your dataset size
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # args related to eval/save
        logging_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        fp16=True,  # we have the model train and eval with fp16 precision
        fp16_full_eval=True,
        optim="adamw_bnb_8bit",  # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
        # report_to = "wandb", # install wand to use this
        # hub_model_id = REPO_ID,
        # push_to_hub = True, # wel'll push the model to hub after each epoch
        # model that was wrapped for QLORA training with peft will not have arguments listed in its signature
        # so we need to pass lable names explicitly to calculate val loss
        label_names=["labels"],
        dataloader_num_workers=4,  # let's get more workers since iterating on video datasets might be slower in general
    )

    trainer = Trainer(
        model=model,
        tokenizer=processor,
        data_collator=LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
    )
