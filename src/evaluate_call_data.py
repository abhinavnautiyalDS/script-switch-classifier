
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["DEVANAGARI", "LATIN"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/evaluation/call_data_final_eval.csv",
    )
    parser.add_argument(
        "--model",
        default="artifacts/bert_tiny_script_classifier",
    )
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    # Training code may save the Hugging Face model in a nested /model folder.
    if not (model_path / "config.json").exists():
        if (model_path / "model" / "config.json").exists():
            model_path = model_path / "model"
        else:
            raise FileNotFoundError(
                f"Cannot find model config.json under {model_path}"
            )

    if not data_path.exists():
        raise FileNotFoundError(f"Call-data CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    print("CSV columns:", df.columns.tolist())
    print("Total rows:", len(df))

    required = {args.text_col, args.label_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required}; found {df.columns.tolist()}. "
            "Pass the correct --text-col and --label-col."
        )

    df = df.copy()
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df[args.label_col] = (
        df[args.label_col].fillna("").astype(str).str.upper().str.strip()
    )

    # Keep only the two classes the model was trained to predict.
    eligible = (
        df[args.text_col].ne("")
        & df[args.label_col].isin(LABELS)
    )
    excluded = df.loc[~eligible, args.label_col].value_counts().to_dict()
    eval_df = df.loc[eligible].copy().reset_index(drop=True)

    if eval_df.empty:
        raise ValueError("No eligible DEVANAGARI/LATIN labeled rows found.")

    print("Eligible rows:", len(eval_df))
    print("Excluded rows by label/empty text:", excluded)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    # Verify class IDs before interpreting predictions.
    id2label = {
        int(k): str(v).upper()
        for k, v in model.config.id2label.items()
    }
    print("Model label mapping:", id2label)

    if set(id2label.values()) != set(LABELS):
        raise ValueError(
            "Model label mapping is not DEVANAGARI/LATIN. "
            "Verify config.json and the training label mapping before evaluation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = eval_df[args.text_col].tolist()
    predictions = []
    confidences = []

    # Batched inference; no training or parameter updates.
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits, dim=-1)

        scores, ids = probs.max(dim=-1)
        predictions.extend(id2label[int(i)] for i in ids.cpu())
        confidences.extend(scores.cpu().tolist())

    eval_df["predicted_label"] = predictions
    eval_df["confidence"] = confidences
    eval_df["correct"] = (
        eval_df[args.label_col] == eval_df["predicted_label"]
    )

    y_true = eval_df[args.label_col]
    y_pred = eval_df["predicted_label"]

    metrics = {
        "evaluation_file": str(data_path),
        "model_path": str(model_path),
        "eligible_rows": len(eval_df),
        "excluded_rows": len(df) - len(eval_df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS,
                     average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
        "confusion_matrix_order": LABELS,
        "device": str(device),
    }

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(
        out_dir / "call_data_final_eval_predictions.csv",
        index=False,
    )
    with open(out_dir / "call_data_final_eval_metrics.json",
              "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nAccuracy:", f"{metrics['accuracy']:.4%}")
    print("Macro-F1:", f"{metrics['macro_f1']:.4%}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0
    ))
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))
    print("\nMisclassified rows:", int((~eval_df["correct"]).sum()))
    print("Results saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()
