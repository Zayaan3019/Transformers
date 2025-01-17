from transformers import BertForSequenceClassification

def get_bert_model(num_labels=2):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model
