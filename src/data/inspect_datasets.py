"""Inspect public Hugging Face dataset schemas and sample records.

Run: python -m src.data.inspect_datasets
"""
from __future__ import annotations

import json
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

DATASETS = [
    "LingoIITGN/COMI-LINGUA",
    "debajyotimaz/trilingual-hinglish-corpus",
]


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def inspect_dataset(dataset_id: str) -> None:
    print("\n" + "=" * 88)
    print(f"DATASET: {dataset_id}")
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as exc:
        print(f"Could not list configs: {type(exc).__name__}: {exc}")
        return

    print(f"Configs: {configs}")
    for config in configs:
        try:
            splits = get_dataset_split_names(dataset_id, config_name=config)
            print(f"\nConfig: {config} | splits: {splits}")
            for split in splits:
                sample = load_dataset(dataset_id, name=config, split=f"{split}[:3]")
                print(f"  Split: {split}")
                print(f"  Features: {safe_json(sample.features)}")
                print(f"  Sample rows: {safe_json(sample[:3])}")
        except Exception as exc:
            print(f"  Could not inspect config={config!r}: {type(exc).__name__}: {exc}")
    print("Reminder: verify the dataset card, licence, attribution, and usage terms manually.")


def main() -> None:
    for dataset_id in DATASETS:
        inspect_dataset(dataset_id)


if __name__ == "__main__":
    main()
