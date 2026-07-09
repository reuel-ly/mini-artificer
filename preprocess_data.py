"""Preprocess Glaive function-calling samples into chat format for SFTTrainer."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from config import POSITIVE_SAMPLE_RATIO

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


def _message_key(sample: dict) -> tuple[str, str, str]:
    messages = sample["messages"]
    return (
        messages[0]["content"],
        messages[1]["content"],
        messages[2]["content"],
    )


def _dedupe(samples: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for sample in samples:
        key = _message_key(sample)
        if key not in seen:
            seen.add(key)
            out.append(sample)
    return out


def _token_length(messages: list[dict], tokenizer: Any) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(ids)


def filter_by_max_length(
    samples: list[dict],
    tokenizer: Any,
    max_length: int,
) -> list[dict]:
    """Drop samples whose full chat template exceeds max_length tokens."""
    return [
        sample
        for sample in samples
        if _token_length(sample["messages"], tokenizer) <= max_length
    ]


def _maybe_filter_by_length(
    samples: list[dict],
    tokenizer: Any | None,
    max_length: int | None,
    label: str,
) -> list[dict]:
    if tokenizer is None or max_length is None:
        return samples
    filtered = filter_by_max_length(samples, tokenizer, max_length)
    dropped = len(samples) - len(filtered)
    if dropped:
        print(f"Dropped {dropped} over-length {label} samples (>{max_length} tokens)")
    return filtered


def _stratified_split(
    samples: list[dict], train_split_ratio: float
) -> tuple[list[dict], list[dict]]:
    split = int(len(samples) * train_split_ratio)
    return samples[:split], samples[split:]


def preprocess_dataset(
    dataset: "Dataset",
    max_samples: int = 10_000,
    train_split_ratio: float = 0.9,
    *,
    tokenizer: Any | None = None,
    max_length: int | None = None,
) -> tuple["Dataset", "Dataset"]:
    """Preprocess rows, then curate a balanced subset for tool-calling SFT.

    Keeps up to 60% function-call samples and 40% negative (no-call) samples,
    capped at ``max_samples`` total. Invalid rows are dropped. Duplicates are
    removed, over-length rows are dropped when ``tokenizer`` and ``max_length``
    are provided, and the curated set is split stratified by class into
    train/eval using ``train_split_ratio`` (default 90/10).
    """
    function_call_samples: list[dict] = []
    strict_negative_samples: list[dict] = []
    fallback_negative_samples: list[dict] = []

    for sample in dataset:
        processed = preprocess_sample(sample)
        if processed is None:
            continue

        assistant_content = processed["messages"][2]["content"]
        if "<functioncall>" in assistant_content:
            function_call_samples.append(processed)
        elif "<functioncall>" not in sample["chat"]:
            strict_negative_samples.append(processed)
        else:
            fallback_negative_samples.append(processed)

    pos_before_dedupe = len(function_call_samples)
    neg_strict_before = len(strict_negative_samples)
    neg_fallback_before = len(fallback_negative_samples)

    function_call_samples = _dedupe(function_call_samples)
    strict_negative_samples = _dedupe(strict_negative_samples)
    fallback_negative_samples = _dedupe(fallback_negative_samples)

    print(
        f"Deduped positives: {pos_before_dedupe} -> {len(function_call_samples)} "
        f"({pos_before_dedupe - len(function_call_samples)} removed)"
    )
    print(
        f"Deduped strict negatives: {neg_strict_before} -> {len(strict_negative_samples)} "
        f"({neg_strict_before - len(strict_negative_samples)} removed)"
    )
    print(
        f"Deduped fallback negatives: {neg_fallback_before} -> {len(fallback_negative_samples)} "
        f"({neg_fallback_before - len(fallback_negative_samples)} removed)"
    )

    function_call_samples = _maybe_filter_by_length(
        function_call_samples, tokenizer, max_length, "positive"
    )
    strict_negative_samples = _maybe_filter_by_length(
        strict_negative_samples, tokenizer, max_length, "strict negative"
    )
    fallback_negative_samples = _maybe_filter_by_length(
        fallback_negative_samples, tokenizer, max_length, "fallback negative"
    )

    random.seed(42)
    random.shuffle(function_call_samples)
    random.shuffle(strict_negative_samples)
    random.shuffle(fallback_negative_samples)

    n_positive = int(max_samples * POSITIVE_SAMPLE_RATIO)
    n_negative = max_samples - n_positive

    curated_pos = function_call_samples[:n_positive]
    curated_neg = strict_negative_samples[:n_negative]
    fallback_used = 0
    if len(curated_neg) < n_negative:
        need = n_negative - len(curated_neg)
        fallback_used = min(need, len(fallback_negative_samples))
        curated_neg.extend(fallback_negative_samples[:fallback_used])

    print(f"Positive samples: {len(curated_pos)}")
    print(f"Negative samples: {len(curated_neg)} ({fallback_used} from fallback pool)")

    random.shuffle(curated_pos)
    random.shuffle(curated_neg)

    train_pos, eval_pos = _stratified_split(curated_pos, train_split_ratio)
    train_neg, eval_neg = _stratified_split(curated_neg, train_split_ratio)

    train_results = train_pos + train_neg
    eval_results = eval_pos + eval_neg
    random.shuffle(train_results)
    random.shuffle(eval_results)

    train_pos_count = sum(
        1 for row in train_results if "<functioncall>" in row["messages"][2]["content"]
    )
    eval_pos_count = sum(
        1 for row in eval_results if "<functioncall>" in row["messages"][2]["content"]
    )
    print(
        f"Train: {len(train_results)} ({train_pos_count} positive, "
        f"{len(train_results) - train_pos_count} negative)"
    )
    print(
        f"Eval: {len(eval_results)} ({eval_pos_count} positive, "
        f"{len(eval_results) - eval_pos_count} negative)"
    )

    from datasets import Dataset

    return Dataset.from_list(train_results), Dataset.from_list(eval_results)
