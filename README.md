# 🦾 Training Toolkit

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Examples](#examples)
7. [Configuration](#configuration)

## Introduction

## Features

### Supported Models

## Installation

The Training Toolkit is made and tested using Python 3.12.

1. Run `pip3 install -r requirements.txt`

- **Note for MacOS**: the toolkit is using `decord` to load video. This library is no longer maintained, and it's incompatible with MacOS.

2. [optional] Create a `.env` file and put your Hugging Face access token in it: `HF_TOKEN = your_token`. This is necessary to use models out of gated repositories such as PaliGemma.

3. [optional] Run `pip3 install flash-attn --no-build-isolation` to enable Flash Attention 2 on CUDA devices.

## Quick Start

```python
from training_toolkit import build_trainer, paligemma_image_preset, image_json_preset

trainer = build_trainer(
    **image_json_preset.with_path("path/to/dataset").as_kwargs(),
    **paligemma_image_preset.as_kwargs()
)

trainer.train()
```

## Usage

There're two primary parts of the Training Toolkit: a `ModelPreset` and a `Data Preset`.

These are dataclasses that contain model settings, training arguments, collators etc. Combined they provide everything necessary to set up a Trainer with reasonable defaults. You can directly access and override all settings before feeding them to a builder.

```python
paligemma_image_preset.use_lora = True
paligemma_image_preset.training_args["per_device_train_batch_size"] = 8
```

The process of building a trainer is handled by the `build_trainer` factory function. It instantiates the model, enables adapters, runs preprocessing on the data and wraps it all into a Trainer. You can can get all the necessary arguments using the `.as_kwargs()` method of the preset, or you can pass them directly.

You can access individual components of a traner directly as its properties.

```python
model = trainer.model
processor = trainer.tokenizer
eval_dataset = trainer.eval_dataset
```

## Examples

Check out the cookbook section of this repository to find walkthroughs for supported usecases.

- How to fine-tune LlaVa-NeXT-Video on a local dataset for QA: [notebook](https://github.com/tensorsense/training_toolkit/blob/main/cookbook/import_video_and_ft_llava.ipynb).
- How to fine-tune PaliGemma on a local image dataset for JSON output: [notebook](https://github.com/tensorsense/training_toolkit/blob/main/cookbook/import_images_and_ft_paligemma.ipynb).

## Configuration

