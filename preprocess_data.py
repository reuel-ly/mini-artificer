"""Preprocess Glaive function-calling samples into chat format for SFTTrainer."""

from __future__ import annotations

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


def preprocess_dataset(dataset: "Dataset") -> "Dataset":
    """Map ``preprocess_sample`` over a dataset and drop invalid rows."""
    processed = dataset.map(
        preprocess_sample,
        remove_columns=dataset.column_names,
    )
    return processed.filter(lambda row: row["messages"] is not None)
