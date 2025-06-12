import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("resume.csv").dropna()
df = df[df['label'].isin([0, 1])]
train_texts, val_texts, train_labels, val_labels = train_test_split(
    list(zip(df["job_description"], df["resume_text"])), df["label"], test_size=0.2
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text_pairs):
    return tokenizer(
        [t[0] for t in text_pairs],
        [t[1] for t in text_pairs],
        padding=True,
        truncation=True,
        max_length=512
    )

# Dataset class
class ResumeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare encodings and datasets
train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)
train_dataset = ResumeDataset(train_encodings, list(train_labels))
val_dataset = ResumeDataset(val_encodings, list(val_labels))

# Define model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Output directory for model saving
save_dir = "backend/model/pretrained_model"
os.makedirs(save_dir, exist_ok=True)

# Training args
training_args = TrainingArguments(
    output_dir=save_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Save model and tokenizer to the new folder
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Model and tokenizer saved to: {save_dir}")
