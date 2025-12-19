import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

# Load model and tokenizer from local path
model_path = "E:\\model\\garak-llm_toxic-comment-model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load input data
input_path = "C:\\Users\\21888\\Desktop\\test\\11.xlsx"
df = pd.read_excel(input_path)
texts = df.iloc[:, 0].tolist()  # Get first column as text inputs

# Prepare for inference
results = []
confidences = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities using softmax
    probs = softmax(outputs.logits, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    
    results.append(predicted.item())
    confidences.append(confidence.item())

# Create result DataFrame
result_df = df.copy()
result_df['toxic_3'] = results
result_df['toxic_confidence_3'] = confidences

# Save results
output_path = "C:\\Users\\21888\\Desktop\\test\\result_toxic_3.xlsx"
result_df.to_excel(output_path, index=False)