"""Download public corpora and prepare script-classification CSV splits.

This script never reads call_data(3).txt. That file is reserved for final
real-world evaluation and must remain outside the training workflow.

Run: python -m src.data.prepare_dataset
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.script_labels import count_scripts

SEED = 42
OUTPUT_DIR = Path("data/processed")
COMI_ID = "LingoIITGN/COMI-LINGUA"
TRILINGUAL_ID = "debajyotimaz/trilingual-hinglish-corpus"


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode for model input and duplicate checks."""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def duplicate_key(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def to_examples(text: Any, source: str, split: str, group_id: str) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        return []
    text = normalize_text(text)
    if not text:
        return []
    counts = count_scripts(text)
    if counts.label not in {"DEVANAGARI", "LATIN"}:
        return []
    return [{
        "text": text,
        "label": counts.label,
        "source": source,
        "source_split": split,
        "group_id": group_id,
        "devanagari_count": counts.devanagari,
        "latin_count": counts.latin,
    }]


def dataset_rows(dataset_id: str, config: str | None = None) -> dict[str, list[dict[str, Any]]]:
    ds = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
    return {split: [dict(row) for row in ds[split]] for split in ds.keys()}


def make_comi_examples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"Loading {COMI_ID} / LID ...")
    splits = dataset_rows(COMI_ID, "LID")
    if "train" not in splits:
        raise ValueError(f"COMI-LINGUA LID has no train split: {list(splits)}")
    train_rows = splits["train"]
    test_rows = splits.get("test", [])
    if train_rows and "Sentences" not in train_rows[0]:
        raise ValueError(f"Unexpected COMI-LINGUA columns: {list(train_rows[0])}")

    rng = random.Random(SEED)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    val_count = max(1, round(len(indices) * 0.10)) if len(indices) > 1 else 0
    val_ids = set(indices[:val_count])
    train_examples: list[dict[str, Any]] = []
    val_examples: list[dict[str, Any]] = []
    test_examples: list[dict[str, Any]] = []

    for i, row in enumerate(train_rows):
        target = val_examples if i in val_ids else train_examples
        target.extend(to_examples(
            row.get("Sentences"), "COMI-LINGUA-LID",
            "validation" if i in val_ids else "train", f"comi-train-{i}"
        ))
    for i, row in enumerate(test_rows):
        test_examples.extend(to_examples(
            row.get("Sentences"), "COMI-LINGUA-LID", "public_test", f"comi-test-{i}"
        ))
    return train_examples + val_examples, test_examples


def make_trilingual_examples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"Loading {TRILINGUAL_ID} ...")
    splits = dataset_rows(TRILINGUAL_ID)
    if "train" not in splits:
        raise ValueError(f"Trilingual corpus has no train split: {list(splits)}")
    train_rows = splits["train"]
    test_rows = splits.get("test", [])
    required = {"english", "hindi", "hinglish"}
    if train_rows and not required.issubset(train_rows[0]):
        raise ValueError(f"Unexpected trilingual columns: {list(train_rows[0])}")

    rng = random.Random(SEED + 1)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    val_count = max(1, round(len(indices) * 0.10)) if len(indices) > 1 else 0
    val_ids = set(indices[:val_count])

    def expand(rows: list[dict[str, Any]], split_name: str, val_indices: set[int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train_part: list[dict[str, Any]] = []
        val_part: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            actual_split = "validation" if val_indices is not None and i in val_indices else split_name
            dest = val_part if actual_split == "validation" else train_part
            group = f"tri-{split_name}-{row.get('id', i)}"
            for field in ("english", "hindi", "hinglish"):
                dest.extend(to_examples(row.get(field), f"TRILINGUAL-{field}", actual_split, group))
        return train_part, val_part

    train_part, val_part = expand(train_rows, "train", val_ids)
    test_part, _ = expand(test_rows, "public_test")
    return train_part + val_part, test_part


def deduplicate_across_splits(
    train: pd.DataFrame, validation: pd.DataFrame, public_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Remove normalized text overlap; keep public_test, then validation, then train."""
    seen: set[str] = set()
    removed: dict[str, int] = {}
    outputs: list[pd.DataFrame] = []
    for name, frame in (("public_test", public_test), ("validation", validation), ("train", train)):
        if frame.empty:
            outputs.append(frame.copy())
            removed[name] = 0
            continue
        keep = []
        for text in frame["text"].tolist():
            key = duplicate_key(text)
            keep.append(key not in seen)
            seen.add(key)
        cleaned = frame.loc[keep].copy().reset_index(drop=True)
        removed[name] = len(frame) - len(cleaned)
        outputs.append(cleaned)
    test_out, val_out, train_out = outputs
    return train_out, val_out, test_out, removed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comi_train_val, comi_test = make_comi_examples()
    tri_train_val, tri_test = make_trilingual_examples()
    all_train_val = comi_train_val + tri_train_val
    all_public_test = comi_test + tri_test

    train_val_df = pd.DataFrame(all_train_val)
    public_test_df = pd.DataFrame(all_public_test)
    if train_val_df.empty:
        raise RuntimeError("No train/validation examples were produced; inspect dataset schemas.")

    val_mask = train_val_df["source_split"].eq("validation")
    train_df = train_val_df.loc[~val_mask].copy()
    validation_df = train_val_df.loc[val_mask].copy()
    train_df, validation_df, public_test_df, removed = deduplicate_across_splits(
        train_df, validation_df, public_test_df
    )

    for name, frame in (("train", train_df), ("validation", validation_df), ("public_test", public_test_df)):
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8")

    stats: dict[str, Any] = {
        "seed": SEED,
        "sources": {
            COMI_ID: "LID/Sentences; source train divided into train/validation; source test kept as public_test",
            TRILINGUAL_ID: "english/hindi/hinglish expanded by row; source train divided by row into train/validation; source test kept as public_test",
        },
        "counts": {},
        "duplicates_removed": removed,
        "label_policy": "Unicode dominant script; only DEVANAGARI/LATIN retained; TIE, OUT_OF_SCOPE, NO_LETTERS skipped",
        "call_data_used": False,
    }
    for name, frame in (("train", train_df), ("validation", validation_df), ("public_test", public_test_df)):
        stats["counts"][name] = {
            "rows": int(len(frame)),
            "labels": {k: int(v) for k, v in frame["label"].value_counts().to_dict().items()} if not frame.empty else {},
            "sources": {k: int(v) for k, v in frame["source"].value_counts().to_dict().items()} if not frame.empty else {},
        }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nPrepared files are in: {OUTPUT_DIR.resolve()}")
    print("call_data(3).txt was not opened or used.")


if __name__ == "__main__":
    main()
