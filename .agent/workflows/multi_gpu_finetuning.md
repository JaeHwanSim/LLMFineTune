---
description: How to fine-tune LLMs using Multi-GPU with Hugging Face Accelerate and DeepSpeed
---

# Multi-GPU Fine-Tuning Guide (without Unsloth)

This guide explains how to utilize multiple GPUs (e.g., 2x RTX 4090) to fine-tune large language models (like Gemma 2 27B) using Hugging Face `transformers`, `peft`, and `accelerate` with `DeepSpeed`. This approach allows you to leverage the combined VRAM of multiple GPUs, enabling the training of larger models that might not fit on a single card.

## 1. Prerequisites

Ensure you have the necessary libraries installed. You will need `deepspeed` for efficient memory management across GPUs.

```bash
pip install transformers peft accelerate bitsandbytes deepspeed trl
```

## 2. Configuration (Accelerate & DeepSpeed)

You need to configure `accelerate` to use DeepSpeed. This is crucial for sharding the model across GPUs to save memory (Zero Redundancy Optimizer - ZeRO).

Run the following command and answer the prompts:

```bash
accelerate config
```

**Recommended Settings for 2x GPUs (ZeRO-2 or ZeRO-3):**

*   **In which compute environment are you running?**: `This machine`
*   **Which type of machine are you using?**: `multi-GPU`
*   **How many different machines will you use?**: `1`
*   **Do you want to use DeepSpeed?**: `yes`
*   **Do you want to specify a json file to a DeepSpeed config?**: `no` (or yes if you have one)
*   **Zero Stage**: `2` (Optimizes optimizer states & gradients) or `3` (Offloads parameters too - best for very large models but slower)
*   **Gradient Accumulation**: `yes` (e.g., 2 or 4)
*   **Gradient Clipping**: `yes` (1.0)
*   **Offload Optimizer to CPU?**: `no` (unless you are extremely low on VRAM)

## 3. Training Script (`train_multigpu.py`)

Create a new python script (e.g., `train_multigpu.py`). Note that we do **not** use `Unsloth` here.

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# 1. Model & Tokenizer Configuration
model_id = "google/gemma-2-27b-it"  # Or your target model

# 4-bit Quantization Config (Crucial for memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right" # Mixed results with left/right, try right for Gemma

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()}, # Important for DeepSpeed/Accelerate
    attn_implementation="flash_attention_2",      # Use Flash Attention 2 if available
    torch_dtype=torch.bfloat16
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# 2. LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Dataset
dataset = load_dataset("json", data_files="ProductData/neobiotech_final_1000.jsonl", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples.get("input", [""] * len(instructions))
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Gemma 2 format
        text = f"<start_of_turn>user\n{instruction} {input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./results_multigpu",
    num_train_epochs=3,
    per_device_train_batch_size=2, # Adjust based on VRAM
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=True, # Enable BF16 for Ampere+ GPUs
    logging_steps=10,
    optim="paged_adamw_8bit", # Saves memory
    save_strategy="epoch",
    ddp_find_unused_parameters=False, # Often needed for DDP
)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024, # Reduce if OOM occurs
    peft_config=peft_config,
    args=training_args,
    packing=False,
)

# 6. Train
trainer.train()

# 7. Save
trainer.save_model("./gemma2_27b_multigpu_adapter")
```

## 4. Execution

Do **not** run with `python train_multigpu.py`. You must use `accelerate launch`.

```bash
accelerate launch train_multigpu.py
```

## Key Differences from Unsloth

1.  **Device Map**: We use `device_map={"": torch.cuda.current_device()}` instead of `auto` or specific maps, as Accelerate/DeepSpeed handles device placement.
2.  **DeepSpeed**: The magic happens via the `accelerate config` which enables DeepSpeed. This shards the model/optimizer states across your GPUs.
3.  **Speed**: It might be slower per-step than Unsloth, but you can potentially use a larger global batch size or train larger models that physically don't fit on one GPU.
