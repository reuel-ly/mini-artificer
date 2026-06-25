"""Load the Glaive function-calling v2 dataset from Hugging Face."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from config import DATASET_NAME

if TYPE_CHECKING:
    from datasets import Dataset

JSON_FILENAME = "glaive-function-calling-v2.json"


def load_glaive_dataset(split: str = "train") -> "Dataset":
    """Load Glaive function-calling v2 from Hugging Face.

    Tries ``datasets.load_dataset`` first. On ImportError (e.g. broken scipy on
    Colab), falls back to downloading the raw JSON via ``huggingface_hub``.
    """
    try:
        from datasets import load_dataset

        return load_dataset(DATASET_NAME)[split]
    except ImportError as e:
        print(f"datasets import failed, using JSON fallback: {e}")
        return _load_from_json()


def _load_from_json() -> "Dataset":
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=JSON_FILENAME,
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return Dataset.from_list(records)


if __name__ == "__main__":
    dataset = load_glaive_dataset()
    print(dataset[0])
    print("---")
    print(dataset[1])
