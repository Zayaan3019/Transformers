# Transformer Fine-Tuning for Text Classification

This project demonstrates fine-tuning a pre-trained BERT model for text classification. It uses the HuggingFace Transformers library and PyTorch to implement the model, training it on a custom text dataset.

## Project Overview
This project fine-tunes BERT (Bidirectional Encoder Representations from Transformers) on a text classification task. The goal is to achieve high accuracy in predicting labels for the given text samples.

## Features
- Fine-tuning BERT for text classification.
- Tokenization and preprocessing using HuggingFace's transformers.
- Evaluation using accuracy metrics.
- Custom data preprocessing pipeline.

## Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Transformer-Fine-Tuning.git
    cd Transformer-Fine-Tuning
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your `train.csv` and `test.csv` dataset in the `data` folder.

## Usage
Run the `main.py` script to fine-tune and evaluate the model:
```bash
python main.py
