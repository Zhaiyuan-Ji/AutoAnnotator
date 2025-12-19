import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax

# Load model and tokenizer from local path
model_path = "E:\\model\\s-nlp_roberta_toxicity_classifier"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Read input data
input_file = "C:\\Users\\21888\\Desktop\\test\\11.xlsx"
df = pd.read_excel(input_file)

# Process each text
results = []
confidences = []
for text in df.iloc[:, 0].astype(str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = softmax(outputs.logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()
    
    results.append(pred_label)
    confidences.append(confidence)

# Save results
df['toxic_1'] = results
df['toxic_confidence_1'] = confidences
output_file = "C:\\Users\\21888\\Desktop\\test\\result_toxic_1.xlsx"
df.to_excel(output_file, index=False)