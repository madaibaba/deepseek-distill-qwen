"""
===============================================================================
    Script for Fine-Tuning Qwen Model with DeepSeek Dataset
===============================================================================

Author:
-------
houalex@gmail.com

Date:
-----
Feb 14, 2025

Description:
-------------
This script fine-tunes the Qwen 2.5B model using the DeepSeek dataset, which contains 
theorems and proof responses generated from DeepSeek. The script:
- Loads the dataset.
- Prepares the data for model fine-tuning.
- Applies low-rank adaptation (LoRA) for efficient training.
- Trains and evaluates the model.
- Saves the fine-tuned model and tokenizer.

Dependencies:
-------------
- datasets (for loading the dataset)
- transformers (for model and tokenizer)
- torch (for deep learning)
- peft (for parameter-efficient fine-tuning)
- trl (for specialized trainer)
- flash-attn (for efficient attention implementation)

Usage:
------
1. Install the required dependencies:
   $ pip install datasets transformers torch peft trl flash-attn

2. Run the script to:
   - Fine-tune the model using DeepSeek dataset.
   - Save the trained model and tokenizer to disk.

License:
--------
Apache License 2.0 (Apache-2.0)

===============================================================================
"""

# -*- coding: utf-8 -*-
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline


# Load the dataset
dataset = load_dataset("json", data_files="./dataset/answers_from_deepseek.json")
dataset = dataset["train"]

# Format the dataset
def format_instruction(example):
    return {
        "text": (
            "<|user|>\n"
            f"{example['question']}\n"
            "<|end|>\n"
            "<|assistant|>\n"
            f"{example['answer']}\n"
            "<|end|>"
        )
    }

# Format the dataset using dataset.column_names
formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=dataset.column_names)

# Train-test split
formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90% training, 10% test

# Print the dataset info
print(formatted_dataset)

# Local Qwen model path
model_path = "./model/Qwen2.5-1.5B"
# Load the local tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Add custom tokens
CUSTOM_TOKENS = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})
tokenizer.pad_token = tokenizer.eos_token

# Load model using flash attention
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
model.resize_token_embeddings(len(tokenizer))  # Adjust to fit custom tokens


# LoRA configuration
peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    bias="none",  # No bias term
    task_type="CAUSAL_LM"  # Task type
)


# Training arguments
training_args = TrainingArguments(
    output_dir="./model/finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    fp16=False,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)


# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    data_collator=data_collator,
    peft_config=peft_config
)

# Start training
trainer.train()
trainer.save_model("./model/finetuned")
tokenizer.save_pretrained("./model/finetuned")


# Final model after merging
final_model = trainer.model.merge_and_unload()
final_model.save_pretrained("./model/deepseek-distill-qwen2.5-1.5b")
tokenizer.save_pretrained("./model/deepseek-distill-qwen2.5-1.5b")
