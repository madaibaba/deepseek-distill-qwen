"""
===============================================================================
    Script for Generating Questions from Lean Theorems and Proofs
===============================================================================

Author:
-------
houalex@gmail.com

Date:
-----
Feb 14, 2025

Description:
-------------
This script loads the 'deepseek-ai/DeepSeek-Prover-V1' dataset from Hugging Face, 
which contains formal Lean theorems and their proof processes. It generates 
questions for each theorem, asking for an analysis, verification, and proof 
validation. The generated questions are saved to a JSON file.

Dependencies:
-------------
- datasets (for loading the Hugging Face dataset)

Usage:
------
1. Install the required dependencies:
   $ pip install datasets

2. Run the script to:
   - Download the dataset
   - Generate the questions
   - Save them to a JSON file

License:
--------
Apache License 2.0 (Apache-2.0)

===============================================================================
"""

# -*- coding: utf-8 -*-
import json
from datasets import load_dataset

# Load dataset from Hugging Face
def load_dataset_from_huggingface():
    dataset_name = 'deepseek-ai/DeepSeek-Prover-V1'  # Dataset name
    dataset = load_dataset(dataset_name)  # Load dataset from Hugging Face
    return dataset['train']  # Assuming the default split is 'train'

# Generate question for the dataset entry
def generate_question(data):
    # Organize the question
    question = {
        "name": data['name'],
        "question": f"""
I have a theorem and a related proof process in Lean, and I would like you to help me analyze it and provide the complete proof. Please verify whether this theorem holds, explain the logic, and provide the proof steps based on the following information.

Here are the details of the theorem:

- **Theorem Name**: `{data['name']}`

- **Theorem Statement**:
    ```lean
    {data['formal_statement']}
    ```

- **Goal**:
    ```lean
    {data['goal']}
    ```

- **Relevant Imports and Setup**:
    ```lean
    {data['header']}
    ```

- **Initial Proof Process**:
    ```lean
    {data['formal_proof']}
    ```

**Please answer the following questions**:
1. Please explain the meaning of the theorem statement and the goal.
2. Does the goal of the theorem hold? If so, please provide the detailed proof process.
3. What is the logical flow of the proof? How should the Lean code be structured?
"""
    }
    return question

# Save questions to a JSON file
def save_questions_to_json(questions, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)

# Main function
def main():
    # Load the Hugging Face dataset
    data = load_dataset_from_huggingface()

    # Generate the list of questions
    questions = []
    for record in data:
        question = generate_question(record)
        questions.append(question)

    # Save questions to a file
    output_file = 'dataset/questions_to_model.json'  # Output file path
    save_questions_to_json(questions, output_file)

    print(f"The list of questions has been saved to {output_file}")

if __name__ == "__main__":
    main()
