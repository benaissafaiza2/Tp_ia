import os
import re
import random
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import torch
from torch import nn

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
def read_csv_safely(path):
    encodings = ["utf-8-sig", "utf-8", "cp1256", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

CSV_PATH = "/root/stage-2025/model/sentiment_multilingual_dataset.csv"  
df = read_csv_safely(CSV_PATH)
print("Loaded shape:", df.shape)
# Make sure the required columns exist
assert {"comment_text", "label"}.issubset(df.columns), "CSV must have columns: comment_text, label"

# Remove superfluous spaces/lines
df["comment_text"] = df["comment_text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()


def normalize_label(x: str) -> str:
    s = x.strip().lower()
    # en
    if s in {"positive", "pos", "positif"}: return "positive"
    if s in {"negative", "neg", "négatif"}: return "negative"
    if s in {"neutral", "neu", "neutre"}:   return "neutral"
    # ar/darija
    if s in {"ايجابي", "إيجابي"}: return "positive"
    if s in {"سلبي"}:             return "negative"
    if s in {"محايد"}:            return "neutral"
    
    if s in {"0", "label_0"}: return "negative"  
    if s in {"1", "label_1"}: return "neutral" 
    if s in {"2", "label_2"}: return "positive"
    return s  

df["label"] = df["label"].map(normalize_label)

# Delete empty rows and duplicates
before = len(df)
df = df.dropna(subset=["comment_text", "label"]).drop_duplicates(subset=["comment_text", "label"])
after = len(df)
print(f"Cleaned: removed {before - after} rows")

# Keep only the three categories
df = df[df["label"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)
print(df["label"].value_counts())
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1,2]),
    y=df["label_id"].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("Class weights:", class_weights.tolist())
train_df, val_df = train_test_split(
    df[["comment_text","label_id"]],
    test_size=0.2,
    stratify=df["label_id"],
    random_state=SEED
)

print("Train size:", len(train_df), "Val size:", len(val_df))
model_name = "xlm-roberta-base"  
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch):
    return tokenizer(
        batch["comment_text"],
        truncation=True,
        padding=False,  
        max_length=160
    )

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["comment_text"])
val_ds   = val_ds.map(tokenize_batch,   batched=True, remove_columns=["comment_text"])


train_ds = train_ds.rename_column("label_id", "labels")
val_ds   = val_ds.rename_column("label_id", "labels")
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
val_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        weights = class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=1e-5,
    num_train_epochs=10,           
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_steps=50,
    seed=SEED,
    fp16=torch.cuda.is_available(),  
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
eval_metrics = trainer.evaluate()
print("Validation metrics:", eval_metrics)
# Detailed taxonomic report
pred = trainer.predict(val_ds)
y_true = pred.label_ids
y_pred = pred.predictions.argmax(-1)
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(3)]))
save_dir = "/root/stage-2025/model/sentiment-analysis_model"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model & tokenizer saved to: {save_dir}")
def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=160,
        return_tensors="pt"
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    return [id2label[p] for p in preds]


samples = [
    "خويا مافهمتش كود ثاني"
 
]
print(predict_sentiment(samples))
