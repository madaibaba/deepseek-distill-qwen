"""
===============================================================================
    Script for Generating Responses using Fine-tuned Qwen Model
===============================================================================

Author:
-------
houalex@gmail.com

Date:
-----
Feb 14, 2025

Description:
-------------
This script loads a fine-tuned Qwen model, which was trained on specific datasets, 
to generate natural language responses to user prompts. The model is used to generate 
text based on a provided question or statement. The generated response is printed 
to the console.

Dependencies:
-------------
- transformers (for loading and using the Qwen model)
- torch (for model inference on GPU/CPU)
- pipeline (for text generation)

Usage:
------
1. Install the required dependencies:
   $ pip install transformers torch

2. Run the script to:
   - Load the fine-tuned Qwen model
   - Generate responses based on the provided prompt
   - Print the generated response

License:
--------
Apache License 2.0 (Apache-2.0)

===============================================================================
"""

# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./model/deepseek-distill-qwen2.5-1.5b",
    torch_dtype=torch.float16,  # Using fp16 for faster inference if possible
    device_map="auto"  # Automatically select the best device (GPU if available)
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/deepseek-distill-qwen2.5-1.5b")
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to match tokenizer size

# Create a text generation pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Generate a response based on a prompt
prompt = """<|user|>
What do you think about Trump?
<|end|>
<|assistant|>
"""

# Generate the response with specified parameters
output = chat_pipeline(
    prompt,
    max_new_tokens=500,  # Limit the output length to 500 tokens
    temperature=0.7,  # Controls randomness of the output (higher is more random)
    do_sample=True,  # Use sampling rather than greedy search
    eos_token_id=tokenizer.eos_token_id  # Ensure the model stops at the end-of-sequence token
)

# Print the generated response
print(output[0]['generated_text'])
