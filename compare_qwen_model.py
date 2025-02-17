"""
===============================================================================
    Model Comparison Script for Qwen Models
===============================================================================

Author:
-------
houalex@gmail.com

Date:
-----
Feb 14, 2025

Description:
-------------
This script compares two versions of the Qwen model (original vs. distilled) on various metrics.
The script measures:
1. Inference Time
2. Memory Usage
3. Perplexity
4. BLEU Score

Dependencies:
-------------
- torch
- transformers
- sacrebleu
- psutil
- tabulate

Usage:
------
1. Place your Qwen model files or use model names from Hugging Face.
2. Run the script to compare the models based on the following metrics:
    - Inference time for processing a prompt.
    - Memory usage during inference.
    - Perplexity score to measure the model's prediction capability.
    - BLEU score to compare the quality of generated text.

Example:
--------
Run the script by calling:
    compare_models('original_qwen_model_path', 'distilled_qwen_model_path')

Note:
-----
Ensure that the models are compatible with Hugging Face's API or are properly loaded.

License:
--------
Apache License 2.0 (Apache-2.0)

===============================================================================
"""

# -*- coding: utf-8 -*-
import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
from tabulate import tabulate

# Function to calculate Perplexity
def calculate_perplexity(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    bleu = sacrebleu.corpus_bleu([candidate], [reference])
    return bleu.score

# Function to compare inference speed and memory usage
def compare_models(model_name_1, model_name_2, prompt="The quick brown fox"):
    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_1)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2)
    
    # Measure inference time
    start_time = time.time()
    _ = model_1.generate(tokenizer(prompt, return_tensors="pt").input_ids)
    end_time = time.time()
    model_1_time = end_time - start_time
    
    start_time = time.time()
    _ = model_2.generate(tokenizer(prompt, return_tensors="pt").input_ids)
    end_time = time.time()
    model_2_time = end_time - start_time
    
    # Measure memory usage
    process = psutil.Process()
    memory_model_1_before = process.memory_info().rss / (1024 * 1024)
    _ = model_1.generate(tokenizer(prompt, return_tensors="pt").input_ids)
    memory_model_1_after = process.memory_info().rss / (1024 * 1024)
    
    memory_model_2_before = process.memory_info().rss / (1024 * 1024)
    _ = model_2.generate(tokenizer(prompt, return_tensors="pt").input_ids)
    memory_model_2_after = process.memory_info().rss / (1024 * 1024)
    
    memory_model_1 = memory_model_1_after - memory_model_1_before
    memory_model_2 = memory_model_2_after - memory_model_2_before

    # Calculate Perplexity
    perplexity_model_1 = calculate_perplexity(model_1, tokenizer, prompt)
    perplexity_model_2 = calculate_perplexity(model_2, tokenizer, prompt)
    
    # Generate sample text
    sample_text_1 = tokenizer.decode(model_1.generate(tokenizer(prompt, return_tensors="pt").input_ids)[0], skip_special_tokens=True)
    sample_text_2 = tokenizer.decode(model_2.generate(tokenizer(prompt, return_tensors="pt").input_ids)[0], skip_special_tokens=True)
    
    # Calculate BLEU score (comparing generated text with reference)
    reference = [prompt]
    bleu_model_1 = calculate_bleu(reference, sample_text_1)
    bleu_model_2 = calculate_bleu(reference, sample_text_2)
    
    # Output comparison results
    table = [
        ["Metric", "Model 1 (Original)", "Model 2 (Distilled)"],
        ["Inference Time (s)", f"{model_1_time:.3f}", f"{model_2_time:.3f}"],
        ["Memory Usage (MB)", f"{memory_model_1:.2f}", f"{memory_model_2:.2f}"],
        ["Perplexity", f"{perplexity_model_1:.2f}", f"{perplexity_model_2:.2f}"],
        ["BLEU Score", f"{bleu_model_1:.2f}", f"{bleu_model_2:.2f}"],
    ]
    
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    
# Call function to compare two models
compare_models('./model/Qwen2.5-1.5B', './model/deepseek-distill-qwen2.5-1.5b')
