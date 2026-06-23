"""Preprocess Glaive function-calling samples into chat format for SFTTrainer."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

USER_RE = re.compile(
    r"USER:\s*(.*?)(?=\n(?:ASSISTANT|FUNCTION RESPONSE|USER):|<\|endoftext\|>|\Z)",
    re.DOTALL,
)
ASSISTANT_RE = re.compile(
    r"ASSISTANT:\s*(.*?)(?=\n(?:FUNCTION RESPONSE|USER|ASSISTANT):|<\|endoftext\|>|\Z)",
    re.DOTALL,
)


def preprocess_sample(sample: dict) -> dict | None:
    """Convert one raw Glaive row into SFTTrainer chat format (Option A).

    Keeps the first USER/ASSISTANT turn only and stops before FUNCTION RESPONSE.
    Returns None for malformed or empty rows.
    """
    chat = sample["chat"].strip()
    user_match = USER_RE.search(chat)
    asst_match = ASSISTANT_RE.search(chat)
    if not user_match or not asst_match:
        return None

    user_content = user_match.group(1).strip()
    assistant_content = asst_match.group(1).strip()
    if not user_content or not assistant_content:
        return None

    return {
        "messages": [
            {"role": "system", "content": sample["system"].replace("SYSTEM: ", "", 1).strip()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def preprocess_dataset(
    dataset: "Dataset", max_samples: int = 10_000, train_split_ratio: float = 0.9
) -> tuple["Dataset", "Dataset"]:
    """Preprocess rows, then curate a balanced subset for tool-calling SFT.

    Keeps up to 80% function-call samples and 20% negative (no-call) samples,
    capped at ``max_samples`` total. Invalid rows are dropped. The curated set
    is shuffled and split into train/eval using ``train_split_ratio`` (default 90/10).
    """
    function_call_samples = []
    negative_samples = []

    for sample in dataset:
        processed = preprocess_sample(sample)
        if processed is None:
            continue

        assistant_content = processed["messages"][2]["content"]
        if "<functioncall>" in assistant_content:
            function_call_samples.append(processed)
        else:
            negative_samples.append(processed)

    random.seed(42)
    random.shuffle(function_call_samples)
    random.shuffle(negative_samples)

    n_positive = int(max_samples * 0.8)
    n_negative = int(max_samples * 0.2)

    results = function_call_samples[:n_positive] + negative_samples[:n_negative]
    random.shuffle(results)

    from datasets import Dataset

    print(f"Positive samples: {len(function_call_samples[:n_positive])}")
    print(f"Negative samples: {len(negative_samples[:n_negative])}")

    split = int(len(results) * train_split_ratio)
    train_data = Dataset.from_list(results[:split])
    eval_data = Dataset.from_list(results[split:])
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    return train_data, eval_data
