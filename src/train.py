"""Fine-tune the approved BERT-Tiny checkpoint for dominant-script classification.

Uses only data/processed/train.csv and validation.csv.
It deliberately does not read public_test.csv or any call_data file.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
OUTPUT_DIR = Path("artifacts/bert_tiny_script_classifier")
LABELS = ["DEVANAGARI", "LATIN"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
MAX_LENGTH = 128


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the dataset-preparation notebook first and ensure "
            "data/processed/train.csv and validation.csv exist."
        )
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[(df["text"] != "") & df["label"].isin(LABELS)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid labeled examples found in {path}")
    return df


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VALIDATION_PATH)

    # Safety check: no normalized text overlap between training and validation.
    normalize = lambda s: " ".join(str(s).casefold().split())
    train_texts = set(train_df["text"].map(normalize))
    val_texts = set(val_df["text"].map(normalize))
    overlap = train_texts & val_texts
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} normalized text overlaps between train and validation. "
            "Resolve leakage before training."
        )

    print("Training rows:", len(train_df), train_df["label"].value_counts().to_dict())
    print("Validation rows:", len(val_df), val_df["label"].value_counts().to_dict())
    print("Normalized text overlap:", len(overlap))
    print("Checkpoint:", MODEL_NAME)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Diagnostic only: the checkpoint is an English uncased tokenizer, so inspect
    # how it tokenizes Devanagari before interpreting validation performance.
    audit_texts = train_df["text"].sample(min(1000, len(train_df)), random_state=SEED).tolist()
    unk_id = tokenizer.unk_token_id
    audit_tokens = tokenizer(
        audit_texts, truncation=True, max_length=MAX_LENGTH, padding=False
    )["input_ids"]
    token_count = sum(len(ids) for ids in audit_tokens)
    unk_count = sum(sum(token_id == unk_id for token_id in ids) for ids in audit_tokens)
    unk_rate = (unk_count / token_count) if token_count else 0.0
    print(f"Tokenizer audit: [UNK] tokens {unk_count}/{token_count} = {unk_rate:.2%}")
    if unk_rate > 0.05:
        print(
            "WARNING: [UNK] rate exceeds 5%. The model may be relying heavily on "
            "unknown-token patterns for script detection. Record this limitation."
        )

    def to_dataset(df: pd.DataFrame) -> Dataset:
        frame = df.copy()
        frame["labels"] = frame["label"].map(LABEL2ID).astype(int)
        return Dataset.from_pandas(frame[["text", "labels"]], preserve_index=False)

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(gold, preds),
            "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(gold, preds, average="weighted", zero_division=0),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_result = trainer.train()
    val_metrics = trainer.evaluate()
    predictions = trainer.predict(val_ds)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    gold_ids = predictions.label_ids

    report = classification_report(
        gold_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(gold_ids, predicted_ids, labels=list(range(len(LABELS))))

    trainer.save_model(str(OUTPUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))

    results = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "labels": LABELS,
        "tokenizer_unk_rate_sample": unk_rate,
        "train_metrics": train_result.metrics,
        "validation_metrics": val_metrics,
        "validation_classification_report": report,
        "validation_confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": LABELS,
        "note": "Validation only. public_test.csv and call_data evaluation files were not read.",
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)

    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2, default=float))
    print("\nValidation classification report:")
    print(classification_report(gold_ids, predicted_ids, target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted; order DEVANAGARI, LATIN):")
    print(matrix)
    print(f"\nSaved model and results to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
