---
library_name: peft
license: other
base_model: /rds/general/user/yx3422/home/Llama-2-7b-chat-hf
tags:
- base_model:adapter:/rds/general/user/yx3422/home/Llama-2-7b-chat-hf
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/rds/general/user/yx3422/home/Llama-2-7b-chat-hf](https://huggingface.co//rds/general/user/yx3422/home/Llama-2-7b-chat-hf) on the D_train_ft_train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.9.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1