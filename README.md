# Deepseek Distillation of Qwen

This project implements the distillation of the Qwen model using Deepseek, aimed at improving model performance and optimizing inference speed. Below are the descriptions and usage details for each script in this project.

## Script List

### 1. `questions_to_model.py`

**Function**: Generates questions from Lean theorems and proofs.

**Description**:
This script loads the dataset `deepseek-ai/DeepSeek-Prover-V1` from Hugging Face, which contains formal Lean theorems and their proof processes. It generates questions for each theorem, asking for analysis, verification, and proof validation. The generated questions are saved to a JSON file.

**Dependencies**:
- datasets

**Usage**:
1. Install dependencies:
   ```bash
   pip install datasets
   ```
2. Run the script to generate questions and save them to a JSON file.

---

### 2. `answers_from_deepseek.py`

**Function**: Interacts with the DeepSeek API to retrieve answers.

**Description**:
This script interacts with the DeepSeek API by loading model configurations (such as API endpoint, authorization token, etc.) from a `model.json` configuration file. It sends queries to the API and prints the returned answers.

**Dependencies**:
- json
- requests

**Usage**:
1. Ensure the `model.json` file is placed in the same directory as the script.
2. Configure the API URL and token in the `model.json`.
3. Run the script to receive answers.

---

### 3. `deepseek_distill_qwen.py`

**Function**: Fine-tunes the Qwen model using the DeepSeek dataset.

**Description**:
This script fine-tunes the Qwen 2.5B model using the DeepSeek dataset, which contains theorems and proof responses from DeepSeek. The script utilizes Low-Rank Adaptation (LoRA) for efficient training and saves the fine-tuned model and tokenizer.

**Dependencies**:
- datasets
- transformers
- torch
- peft
- trl
- flash-attn

**Usage**:
1. Install dependencies:
   ```bash
   pip install datasets transformers torch peft trl flash-attn
   ```
2. Run the script to fine-tune the model and save the trained model and tokenizer.

---

### 4. `test_qwen_model.py`

**Function**: Generates responses using a fine-tuned Qwen model.

**Description**:
This script loads a fine-tuned Qwen model and generates natural language responses based on provided prompts. The generated responses are printed to the console.

**Dependencies**:
- transformers
- torch

**Usage**:
1. Install dependencies:
   ```bash
   pip install transformers torch
   ```
2. Run the script to generate and print responses.

---

### 5. `compare_qwen_model.py`

**Function**: Compares two Qwen models (original vs distilled) on various metrics.

**Description**:
This script compares two Qwen models (original vs distilled) based on several metrics, including inference time, memory usage, perplexity, and BLEU score.

**Dependencies**:
- torch
- transformers
- sacrebleu
- psutil
- tabulate

**Usage**:
1. Use either Hugging Face models or local model paths to compare models.
2. Run the script to compare models based on the metrics.

**Example**:
```python
compare_models('original_qwen_model_path', 'distilled_qwen_model_path')
```

| **Metric**                   | **Model 1 (Original)** | **Model 2 (Ollama Distilled)** | **Model 2 (250K Distilled)** | **Model 2 (27K Distilled)** |
|------------------------------|-----------------------|------------------------------|-----------------------------|-----------------------------|
| **Inference Time (s)**        | 52.223                | 0.27                         | 0.809                       | 0.746                       |
| **CPU Memory Usage (MB)**     | 0.01                  | 0                            | 0                           | 0                           |
| **GPU Memory Usage (MB)**     | 0.02                  | 0                            | 0                           | 0                           |
| **Perplexity**                | 5.72                  | 40.76                        | 11.45                       | 6.13                        |
| **BLEU Score**                | 0.69                  | 45.63                        | 19.38                       | 21.89                       |
| **ROUGE-1 Score**             | 0.02                  | 0.67                         | 0.4                         | 0.43                        |
| **ROUGE-2 Score**             | 0.01                  | 0.64                         | 0.37                        | 0.4                         |
| **ROUGE-L Score**             | 0.02                  | 0.67                         | 0.4                         | 0.43                        |
| **Model Size (M Parameters)** | 1543.71               | 1777.09                      | 1543.3                      | 1543.3                      |
| **Throughput (samples/sec)**  | 0.02                  | 3.99                         | 1.22                        | 1.3                         |

---

## Installation & Usage

### Install Dependencies

Install the required dependencies for all scripts:

```bash
pip install -r requirements.txt
```

### Running the Scripts

Refer to each script's description above for specific instructions on how to run them. Make sure to install all dependencies and configure the necessary paths and files.

## Contributing

Contributions are welcome! You can participate by following these steps:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## License

This project is licensed under the terms of the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgments

- Thanks to the Deepseek team for their contributions to model distillation.
- Thanks to the authors of the Qwen model for their outstanding work in the field.
