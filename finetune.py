import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler,
    set_seed,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to data.jsonl")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=3e-5)
args = parser.parse_args()

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "./model"
BATCH_SIZE = 16
MAX_LENGTH = 256

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        label = self.label_encoder.transform([item["label"]])[0]
        encoding["labels"] = torch.tensor(label)
        return encoding

with open(args.data, "r") as f:
    raw_data = [json.loads(line.strip()) for line in f]

texts = [item["text"] for item in raw_data]
labels = [item["label"] for item in raw_data]

label_encoder = LabelEncoder()
label_encoder.fit(labels)
num_labels = len(label_encoder.classes_)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

dataset = SentimentDataset(raw_data, tokenizer, label_encoder)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=args.lr)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(loader) * args.epochs,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

print("Starting training...")
for epoch in range(args.epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

label_map = {f"LABEL_{i}": label for i, label in enumerate(label_encoder.classes_)}
with open(os.path.join(SAVE_DIR, "labels.json"), "w") as f:
    json.dump(label_map, f)