import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import os

gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


# 加载数据
data = pd.read_excel('C:\\Users\\21888\\Desktop\\test\\finetune_toxic.xlsx')
texts = data['text'].tolist()
labels = data['label'].tolist()


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)


train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})


model_name = 'E:\\model\\s-nlp_roberta_toxicity_classifier'
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=1000,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.save_model(model_name)

tokenizer.save_pretrained(model_name)