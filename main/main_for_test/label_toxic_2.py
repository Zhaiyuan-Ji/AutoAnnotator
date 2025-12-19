import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.nn.functional import softmax

# Load model and tokenizer from local path
model_path = "E:\\model\\JungleLee_bert-toxic-comment-classification"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Read input data
input_path = 'C:\\Users\\21888\\Desktop\\test\\11.xlsx'
df = pd.read_excel(input_path)

# Prepare for inference
results = []
confidences = []

for text in df.iloc[:, 0].astype(str):  # first column
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item()
    
    results.append(pred)
    confidences.append(confidence)

# Add results to dataframe
df['toxic_2'] = results
df['toxic_confidence_2'] = confidences

# Save results
output_path = 'C:\\Users\\21888\\Desktop\\test\\result_toxic_2.xlsx'
df.to_excel(output_path, index=False)