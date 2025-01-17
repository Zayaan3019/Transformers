import pandas as pd
from tokenizer import get_tokenizer

def preprocess_data(file_path):
    tokenizer = get_tokenizer()
    data = pd.read_csv(file_path)
    encodings = tokenizer(list(data['text']), truncation=True, padding=True, max_length=512)
    return encodings, list(data['label'])
