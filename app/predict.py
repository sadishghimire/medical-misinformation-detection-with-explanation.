# predict.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
#MODEL_PATH = "../saved_model"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")

print(f"Resolved model path: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    # print(f"Input: {text}")
    # print(f"Label: {label.item()}, Confidence: {confidence.item()}")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, label = torch.max(probs, dim=1)
    return label.item(), confidence.item()



