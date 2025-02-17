"""
===============================================================================
    Script for Interacting with the DeepSeek API
===============================================================================

Author:
-------
houalex@gmail.com

Date:
-----
Feb 14, 2025

Description:
-------------
This script interacts with the DeepSeek API to submit queries and retrieve answers.
The model configuration, including API endpoint, authorization token, and other settings,
are loaded from a JSON configuration file ('model.json').

The script does the following:
1. Loads model configuration from 'model.json'.
2. Sends a query to the DeepSeek API using the loaded configuration.
3. Receives and prints the answer from the API.

Dependencies:
-------------
- json
- requests

Usage:
------
1. Place your 'model.json' in the same directory as the script.
2. Ensure that the API URL and token are correctly configured in 'model.json'.
3. Run the script to see the answer returned by the DeepSeek API.

Example:
--------
Run the script to ask a question such as:
"Explain the Lean theorem process step by step."

Note:
-----
Ensure your API token is kept secure and not shared publicly.

License:
--------
Apache License 2.0 (Apache-2.0)

===============================================================================
"""

# -*- coding: utf-8 -*-
import json
import requests

# Read JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Ask DeepSeek and get the answer
def ask_deepseek(query, model_config):
    url = model_config["url"]  # DeepSeek API URL from model.json
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {model_config['token']}"  # Bearer Token from model.json
    }

    data = {
        "inputs": model_config["input"],  # Inputs from model.json
        "query": query,
        "response_mode": model_config["response_mode"],  # Response mode from model.json
        "conversation_id": "",
        "user": model_config["user"],  # User from model.json
        "files": []
    }

    # Send POST request to DeepSeek API
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raises exception for non-2xx responses

        # Process response
        response_data = response.json()
        answer = response_data.get('answer', 'No answer returned')  # Extract 'answer' field
        return answer
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

# Save questions and answers to JSON file
def save_answers_to_json(answers, output_file_path):
    with open(output_file_path, 'a', encoding='utf-8') as file:
        json.dump(answers, file, ensure_ascii=False, indent=4)

# Main function
def main():
    input_file = 'dataset/questions_to_model.json'  # Path to the JSON file with questions
    output_file = 'dataset/answers_from_deepseek.json'  # Path to save answers
    model_file = 'conf/model.json'  # Path to the model.json file with configuration
    
    # Load model configuration from model.json
    model_config = read_json(model_file)
    
    # Read questions
    questions_data = read_json(input_file)
    
    # Store questions and answers
    all_answers = []

    # Iterate through questions, ask DeepSeek, and get answers
    for idx, record in enumerate(questions_data):
        question = record['question']
        
        # Ask DeepSeek for the answer
        answer = ask_deepseek(question, model_config)
        
        # Store the question and answer
        answer_data = {
            "question": question,
            "answer": answer
        }
        
        all_answers.append(answer_data)

        save_answers_to_json(answer_data, output_file)
        
        print(f"Question {idx + 1} has been asked, and the answer has been saved.")
    
    print(f"All questions and answers have been saved to {output_file}")

if __name__ == "__main__":
    main()
