import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from models.bert_model import get_bert_model
from utils.data_preprocessing import preprocess_data
from utils.model_evaluation import evaluate_model

# Load and preprocess data
train_encodings, train_labels = preprocess_data('data/train.csv')
test_encodings, test_labels = preprocess_data('data/test.csv')

# Define DataLoader
train_loader = DataLoader(list(zip(train_encodings, train_labels)), batch_size=16, shuffle=True)
test_loader = DataLoader(list(zip(test_encodings, test_labels)), batch_size=16, shuffle=False)

# Initialize model, optimizer, and loss function
model = get_bert_model(num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train model (simplified training loop)
for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate model
evaluate_model(model, test_loader)
