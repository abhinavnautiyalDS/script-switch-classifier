
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# Paths
ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "reports" / "call_data_final_eval_predictions.csv"
OUT = ROOT / "reports" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["DEVANAGARI", "LATIN"]

# Load final evaluation predictions
df = pd.read_csv(INPUT)

required = {"utterance", "label", "predicted_label"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Evaluate only supported binary classes.
# Keep the original CSV unchanged.
df["label"] = df["label"].astype(str).str.upper().str.strip()
df["predicted_label"] = (
    df["predicted_label"].astype(str).str.upper().str.strip()
)

eligible = df[
    df["label"].isin(LABELS)
    & df["predicted_label"].isin(LABELS)
    & df["utterance"].notna()
    & df["utterance"].astype(str).str.strip().ne("")
].copy()

if eligible.empty:
    raise ValueError("No eligible DEVANAGARI/LATIN rows found.")

y_true = eligible["label"]
y_pred = eligible["predicted_label"]

# -------------------------------
# 1. Overall metrics
# -------------------------------
metrics = {
    "evaluation_rows": int(len(eligible)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_precision": float(
        precision_score(y_true, y_pred, labels=LABELS,
                        average="macro", zero_division=0)
    ),
    "macro_recall": float(
        recall_score(y_true, y_pred, labels=LABELS,
                     average="macro", zero_division=0)
    ),
    "macro_f1": float(
        f1_score(y_true, y_pred, labels=LABELS,
                 average="macro", zero_division=0)
    ),
}

with open(OUT / "overall_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Save detailed per-class report
report = classification_report(
    y_true, y_pred, labels=LABELS,
    output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    OUT / "classification_report.csv"
)

# -------------------------------
# 2. Overall metrics chart
# -------------------------------
names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
values = [
    metrics["accuracy"],
    metrics["macro_precision"],
    metrics["macro_recall"],
    metrics["macro_f1"],
]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(names, values)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Overall Model Evaluation Metrics")
ax.bar_label(bars, labels=[f"{v:.2%}" for v in values], padding=3)
ax.tick_params(axis="x", rotation=12)
fig.tight_layout()
fig.savefig(OUT / "overall_metrics.png", dpi=200)
plt.close(fig)

# -------------------------------
# 3. Confusion matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS,
)
display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Confusion Matrix — Final Call Evaluation")

fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

# -------------------------------
# 4. Per-class precision, recall, F1
# -------------------------------
class_report = classification_report(
    y_true, y_pred, labels=LABELS,
    output_dict=True, zero_division=0
)

class_names = ["DEVANAGARI", "LATIN"]
metric_names = ["precision", "recall", "f1-score"]

fig, ax = plt.subplots(figsize=(9, 5))
x = list(range(len(class_names)))
width = 0.24

for i, metric in enumerate(metric_names):
    vals = [class_report[c][metric] for c in class_names]
    positions = [p + (i - 1) * width for p in x]
    bars = ax.bar(positions, vals, width, label=metric.title())
    ax.bar_label(
        bars,
        labels=[f"{v:.3f}" for v in vals],
        padding=2,
        fontsize=8,
    )

ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Precision, Recall and F1 by Class")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "per_class_metrics.png", dpi=200)
plt.close(fig)

# -------------------------------
# 5. Accuracy by script profile
# -------------------------------
if "script_profile" in eligible.columns:
    profile = eligible.groupby("script_profile").agg(
        rows=("label", "size"),
        accuracy=("correct", "mean")
        if "correct" in eligible.columns
        else ("label", lambda s: 0.0),
    )

    # Recalculate accuracy from actual predictions, avoiding
    # dependence on the pre-existing 'correct' column.
    profile = eligible.assign(
        _correct=eligible["label"] == eligible["predicted_label"]
    ).groupby("script_profile").agg(
        rows=("label", "size"),
        accuracy=("_correct", "mean"),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(profile.index.astype(str), profile["accuracy"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Script Profile")
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(
        bars,
        labels=[
            f"{v:.1%}\n(n={n})"
            for v, n in zip(profile["accuracy"], profile["rows"])
        ],
        padding=3,
    )
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_script_profile.png", dpi=200)
    plt.close(fig)

# -------------------------------
# 6. Accuracy by utterance length
# -------------------------------
if "is_short_le_20_chars" in eligible.columns:
    short = eligible["is_short_le_20_chars"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    length_df = eligible.assign(
        length_group=short.map({
            True: "Short (≤20 characters)",
            False: "Long (>20 characters)",
        }),
        _correct=eligible["label"] == eligible["predicted_label"],
    )

    length_stats = length_df.groupby("length_group").agg(
        rows=("label", "size"),
        accuracy=("_correct", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        length_stats.index,
        length_stats["accuracy"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Utterance Length")
    ax.bar_label(
        bars,
        labels=[
            f"{v:.1%}\n(n={n})"
            for v, n in zip(
                length_stats["accuracy"], length_stats["rows"]
            )
        ],
        padding=3,
    )
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_length.png", dpi=200)
    plt.close(fig)

# -------------------------------
# 7. Confidence: correct vs incorrect
# -------------------------------
if "confidence" in eligible.columns:
    eligible["confidence"] = pd.to_numeric(
        eligible["confidence"], errors="coerce"
    )
    eligible["_correct"] = (
        eligible["label"] == eligible["predicted_label"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for is_correct, name in [
        (True, "Correct predictions"),
        (False, "Incorrect predictions"),
    ]:
        values = eligible.loc[
            eligible["_correct"] == is_correct, "confidence"
        ].dropna()

        if not values.empty:
            ax.hist(values, bins=20, alpha=0.65, label=name)

    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Number of utterances")
    ax.set_title("Confidence Distribution: Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "confidence_distribution.png", dpi=200)
    plt.close(fig)

print("\nOverall evaluation metrics")
for name, value in metrics.items():
    print(
        f"{name}: {value:.4f}"
        if isinstance(value, float)
        else f"{name}: {value}"
    )

print(f"\nCharts and reports saved to: {OUT}")
