"""
===============================================================================
    Model Comparison Script for Qwen Models (Original vs. Distilled)
===============================================================================

Author:
-------
houalex@gmail.com

Date:
-----
Feb 14, 2025

Description:
-------------
This Python script compares two versions of the Qwen model (original vs. distilled) across multiple performance metrics.
The script evaluates models on the following aspects:
1. Inference Time (average time taken to process a prompt)
2. Memory Usage (memory consumed during inference)
3. Perplexity (model's prediction capability)
4. BLEU Score (quality of generated text)
5. F1 Score (classification performance, if applicable)
6. Model Size (number of model parameters in millions)
7. Throughput (samples processed per second)

Dependencies:
-------------
- torch: For loading and running the models.
- transformers: For model and tokenizer management.
- sacrebleu: For calculating BLEU score for text generation.
- psutil: For measuring memory usage during inference.
- tabulate: For formatting the output into a readable table.
- sklearn: For calculating F1 score (if applicable).
- rouge_score: For calculating ROUGE score (optional, if needed).

Usage:
------
1. Replace `'original_qwen_model_path'` and `'distilled_qwen_model_path'` with the paths to your models or use model names from Hugging Face.
2. Run the script to compare the models based on the listed metrics.
3. The script will output the performance comparison in a formatted table.

Example:
--------
To compare the models, run the following:
    compare_models('./model/Qwen2.5-1.5B', './model/deepseek-distill-qwen2.5-1.5b')

Outputs:
--------
The script prints a table comparing both models on the following metrics:
- Inference Time (s)
- Memory Usage (MB)
- Perplexity
- BLEU Score
- F1 Score
- Model Size (M parameters)
- Throughput (samples/sec)

Notes:
-----
- Ensure that your models are compatible with Hugging Face's API or have been properly loaded into the script.
- For F1 Score and ROUGE calculations, ensure you have the relevant labeled data (for F1 score, this is useful in classification tasks).

License:
--------
This script is licensed under the Apache License 2.0 (Apache-2.0).

===============================================================================
"""

# -*- coding: utf-8 -*-
import time
import torch
import psutil
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate Perplexity
def calculate_perplexity(model, tokenizer, prompt):
    """
    Calculate the perplexity of the model on a given prompt.
    Perplexity measures how well the model predicts the next token.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    """
    Calculate the BLEU score, which evaluates the quality of the generated text.
    BLEU score compares the n-grams between the candidate and the reference text.
    """
    bleu = sacrebleu.corpus_bleu([candidate], [reference])
    return bleu.score

# Function to calculate F1 Score
def calculate_f1_score(predictions, labels):
    """
    Calculate the F1 score for classification tasks, combining precision and recall.
    """
    return f1_score(labels, predictions, average='weighted')

# Function to calculate ROUGE score
def calculate_rouge(reference, candidate):
    """
    Calculate the ROUGE score for text summarization.
    ROUGE evaluates the overlap between the reference and candidate n-grams.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    # Return the individual scores for each metric
    return {metric: score.fmeasure for metric, score in scores.items()}

# Function to calculate model size (in millions of parameters)
def get_model_size(model):
    """
    Calculate the size of the model in millions of parameters.
    """
    return sum(p.numel() for p in model.parameters()) / 1e6

# Function to measure inference time (time per prompt)
def measure_inference_time(model, tokenizer, prompt, iterations=5):
    """
    Measure the average inference time for processing a prompt.
    """
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        _ = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        total_time += (time.time() - start_time)
    return total_time / iterations

# Function to measure CPU memory usage
def get_memory_usage(model, tokenizer, prompt, iterations=5):
    """
    Measure the CPU memory usage before and after processing the prompt.
    """
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    for _ in range(iterations):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        _ = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    return memory_after - memory_before

# Function to measure GPU memory usage
def get_gpu_memory_usage(model, tokenizer, prompt, iterations=5):
    """
    Measure the GPU memory usage before and after processing the prompt.
    Returns 'N/A' if no GPU is available.
    """
    if not torch.cuda.is_available():
        return "N/A"  # No GPU available
    
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
    for _ in range(iterations):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        _ = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    torch.cuda.synchronize()  # Synchronize to ensure memory usage is updated
    memory_after = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
    return memory_after - memory_before

# Function to calculate throughput (samples per second)
def calculate_throughput(model, tokenizer, prompt, batch_size=32):
    """
    Calculate the throughput (number of samples processed per second) during inference.
    """
    start_time = time.time()
    for _ in range(batch_size):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        _ = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    end_time = time.time()
    return batch_size / (end_time - start_time)

# Function to compare two models and display the performance metrics
def compare_models(model_name_1, model_name_2, prompt="The quick brown fox"):
    """
    Compare two models on several performance metrics and display the results.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_1)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1).to(device)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2).to(device)

    # Measure Inference Time
    model_1_time = measure_inference_time(model_1, tokenizer, prompt)
    model_2_time = measure_inference_time(model_2, tokenizer, prompt)

    # Measure CPU Memory Usage
    memory_model_1 = get_memory_usage(model_1, tokenizer, prompt)
    memory_model_2 = get_memory_usage(model_2, tokenizer, prompt)

    # Measure GPU Memory Usage
    gpu_memory_model_1 = get_gpu_memory_usage(model_1, tokenizer, prompt)
    gpu_memory_model_2 = get_gpu_memory_usage(model_2, tokenizer, prompt)

    # Calculate Perplexity
    perplexity_model_1 = calculate_perplexity(model_1, tokenizer, prompt)
    perplexity_model_2 = calculate_perplexity(model_2, tokenizer, prompt)

    # Generate text samples for BLEU and ROUGE calculation
    inputs_1 = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    inputs_2 = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    # Ensure input tensors are properly transferred to the device
    sample_text_1 = tokenizer.decode(model_1.generate(inputs_1['input_ids'], attention_mask=inputs_1.get('attention_mask'))[0], skip_special_tokens=True)
    sample_text_2 = tokenizer.decode(model_2.generate(inputs_2['input_ids'], attention_mask=inputs_2.get('attention_mask'))[0], skip_special_tokens=True)

    # Calculate BLEU score
    reference = [prompt]
    bleu_model_1 = calculate_bleu(reference, sample_text_1)
    bleu_model_2 = calculate_bleu(reference, sample_text_2)

    # Calculate ROUGE score
    rouge_model_1 = calculate_rouge(prompt, sample_text_1)
    rouge_model_2 = calculate_rouge(prompt, sample_text_2)

    # Calculate Model Size
    model_size_1 = get_model_size(model_1)
    model_size_2 = get_model_size(model_2)

    # Calculate Throughput
    throughput_model_1 = calculate_throughput(model_1, tokenizer, prompt)
    throughput_model_2 = calculate_throughput(model_2, tokenizer, prompt)

    # Create a comparison table
    table = [
        ["Metric", "Model 1 (Original)", "Model 2 (Distilled)"],
        ["Inference Time (s)", f"{model_1_time:.3f}", f"{model_2_time:.3f}"],
        ["CPU Memory Usage (MB)", f"{memory_model_1:.2f}", f"{memory_model_2:.2f}"],
        ["GPU Memory Usage (MB)", f"{gpu_memory_model_1:.2f}", f"{gpu_memory_model_2:.2f}"],
        ["Perplexity", f"{perplexity_model_1:.2f}", f"{perplexity_model_2:.2f}"],
        ["BLEU Score", f"{bleu_model_1:.2f}", f"{bleu_model_2:.2f}"],
        ["ROUGE-1 Score", f"{rouge_model_1['rouge1']:.2f}", f"{rouge_model_2['rouge1']:.2f}"],
        ["ROUGE-2 Score", f"{rouge_model_1['rouge2']:.2f}", f"{rouge_model_2['rouge2']:.2f}"],
        ["ROUGE-L Score", f"{rouge_model_1['rougeL']:.2f}", f"{rouge_model_2['rougeL']:.2f}"],
        ["Model Size (M Parameters)", f"{model_size_1:.2f}", f"{model_size_2:.2f}"],
        ["Throughput (samples/sec)", f"{throughput_model_1:.2f}", f"{throughput_model_2:.2f}"]
    ]

    print(tabulate(table, headers="firstrow", tablefmt="grid"))

# Call function to compare two models
# Example usage with local models
compare_models('./model/Qwen2.5-1.5B', './model/deepseek-distill-qwen2.5-1.5b', prompt="The quick brown fox jumped over the lazy dog.")

