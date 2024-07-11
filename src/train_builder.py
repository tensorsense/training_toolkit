from transformers import BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch


def build_trainer(
    train_dataset,
    test_dataset,
    collator_cls,
    hf_model_id,
    hf_model_cls,
    hf_processor_cls,
    use_qlora=True,
    use_lora=False,
    lora_target_modules=None,
    training_args=None,
    preprocessor_cls=None,
    preprocessor_kwargs=None,
    **kwargs,
):
    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
    # Standard LoRA:  model is loaded with standard LoRA adaptations.
    # Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

    # And we also need to load the processor for collate_fn
    processor = hf_processor_cls.from_pretrained(hf_model_id, use_fast=False)
    processor.tokenizer.padding_side = (
        "right"  # during training, one always uses padding on the right
    )

    if preprocessor_cls is not None:
        preprocessor = preprocessor_cls(processor, **preprocessor_kwargs)
        train_dataset = preprocessor(train_dataset, split="train")
        test_dataset = preprocessor(test_dataset, split="test")

    print(len(train_dataset))
    print(len(test_dataset))

    if use_qlora or use_lora:
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = hf_model_cls.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=lora_target_modules,
            init_lora_weights="gaussian",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = hf_model_cls.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto",
        )

    trainer = Trainer(
        model=model,
        tokenizer=processor,
        data_collator=collator_cls(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=TrainingArguments(**training_args),
    )

    return trainer
