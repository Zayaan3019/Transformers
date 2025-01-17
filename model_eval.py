from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'])
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
