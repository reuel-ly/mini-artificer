from data_loader import load_glaive_dataset
from preprocess_data import (
    ASSISTANT_RE,
    USER_RE,
    _dedupe,
    filter_by_max_length,
    preprocess_dataset,
    preprocess_sample,
)


def assert_clean_result(result: dict, *, expect_functioncall: bool) -> None:
    assert result is not None
    assert len(result["messages"]) == 3
    assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant"]
    assert not result["messages"][0]["content"].startswith("SYSTEM: ")
    for msg in result["messages"]:
        assert "<|endoftext|>" not in msg["content"]
    assistant_content = result["messages"][-1]["content"]
    if expect_functioncall:
        assert "<functioncall>" in assistant_content
    else:
        assert "<functioncall>" not in assistant_content


def print_result(label: str, result: dict) -> None:
    print(f"=== {label} ===")
    for msg in result["messages"]:
        print(f"[{msg['role'].upper()}]")
        print(msg["content"])
        print()


def test_raw_regex() -> None:
    print("=== Raw regex extraction (intentionally dirty) ===")

    test1 = "USER: Can you book a flight?\n\nASSISTANT: I'm sorry, I can't. <|endoftext|>"
    test2 = (
        "USER: US news?\n\n"
        'ASSISTANT: <functioncall> {"name": "get_news"} <|endoftext|>\n\n'
        'FUNCTION RESPONSE: {"headlines": []}\n\n'
        "USER: France news?"
    )
    test3 = "USER: Hello?\n\n"

    for i, test in enumerate([test1, test2, test3], 1):
        u = USER_RE.search(test)
        a = ASSISTANT_RE.search(test)
        print(f"Case {i}:")
        print(f"  USER:      {u.group(1).strip() if u else 'NO MATCH'}")
        print(f"  ASSISTANT: {a.group(1).strip() if a else 'NO MATCH'}")
        print()


def test_preprocess_edge_cases() -> None:
    print("=== preprocess_sample edge cases ===")

    test1 = "USER: Can you book a flight?\n\nASSISTANT: I'm sorry, I can't. <|endoftext|>"
    test2 = (
        "USER: US news?\n\n"
        'ASSISTANT: <functioncall> {"name": "get_news"} <|endoftext|>\n\n'
        'FUNCTION RESPONSE: {"headlines": []}\n\n'
        "USER: France news?"
    )
    test3 = "USER: Hello?\n\n"

    result1 = preprocess_sample({"system": "SYSTEM: You are helpful.", "chat": test1})
    assert_clean_result(result1, expect_functioncall=False)
    print_result("Edge case 1", result1)

    result2 = preprocess_sample({"system": "SYSTEM: Tools available.", "chat": test2})
    assert_clean_result(result2, expect_functioncall=True)
    assert "FUNCTION RESPONSE" not in result2["messages"][-1]["content"]
    print_result("Edge case 2", result2)

    result3 = preprocess_sample({"system": "SYSTEM: x", "chat": test3})
    assert result3 is None
    print("Edge case 3: preprocess_sample returned None (expected)")
    print()


def test_preprocess_dataset_balances_positive_negative() -> None:
    from datasets import Dataset

    rows = []
    for i in range(8):
        rows.append(
            {
                "system": "SYSTEM: Tools available.",
                "chat": (
                    f"USER: US news topic {i}?\n\n"
                    f'ASSISTANT: <functioncall> {{"name": "get_news_{i}"}} <|endoftext|>'
                ),
            }
        )
    for i in range(4):
        rows.append(
            {
                "system": "SYSTEM: You are helpful.",
                "chat": (
                    f"USER: Can you book flight {i}?\n\n"
                    f"ASSISTANT: I'm sorry, I can't book flight {i}. <|endoftext|>"
                ),
            }
        )

    train_ds, eval_ds = preprocess_dataset(Dataset.from_list(rows), max_samples=10)

    assert len(train_ds) + len(eval_ds) == 10
    assert len(train_ds) == 8
    assert len(eval_ds) == 2
    combined = list(train_ds) + list(eval_ds)
    positive_count = sum(
        1 for row in combined if "<functioncall>" in row["messages"][2]["content"]
    )
    negative_count = len(combined) - positive_count
    assert positive_count == 6
    assert negative_count == 4


def test_preprocess_dataset_train_eval_split() -> None:
    from datasets import Dataset

    rows = []
    for i in range(16):
        rows.append(
            {
                "system": "SYSTEM: Tools available.",
                "chat": (
                    f"USER: US news topic {i}?\n\n"
                    f'ASSISTANT: <functioncall> {{"name": "get_news_{i}"}} <|endoftext|>'
                ),
            }
        )
    for i in range(8):
        rows.append(
            {
                "system": "SYSTEM: You are helpful.",
                "chat": (
                    f"USER: Can you book flight {i}?\n\n"
                    f"ASSISTANT: I'm sorry, I can't book flight {i}. <|endoftext|>"
                ),
            }
        )

    train_ds, eval_ds = preprocess_dataset(Dataset.from_list(rows), max_samples=20)

    assert len(train_ds) + len(eval_ds) == 20
    assert len(train_ds) == 17
    assert len(eval_ds) == 3


def test_preprocess_dataset_drops_invalid_rows() -> None:
    from datasets import Dataset

    test1 = "USER: Can you book a flight?\n\nASSISTANT: I'm sorry, I can't. <|endoftext|>"
    test2 = (
        "USER: US news?\n\n"
        'ASSISTANT: <functioncall> {"name": "get_news"} <|endoftext|>\n\n'
        'FUNCTION RESPONSE: {"headlines": []}\n\n'
        "USER: France news?"
    )
    test3 = "USER: Hello?\n\n"

    ds = Dataset.from_list(
        [
            {"system": "SYSTEM: You are helpful.", "chat": test1},
            {"system": "SYSTEM: Tools available.", "chat": test2},
            {"system": "SYSTEM: x", "chat": test3},
        ]
    )

    train_ds, eval_ds = preprocess_dataset(ds)

    assert len(train_ds) + len(eval_ds) == 2
    for row in list(train_ds) + list(eval_ds):
        assert row["messages"] is not None
        assert len(row["messages"]) == 3
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]


def test_dedupe_removes_duplicate_messages() -> None:
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ]
    }
    deduped = _dedupe([sample, sample, sample])
    assert len(deduped) == 1


def test_filter_by_max_length_drops_long_samples() -> None:
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    short = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    }
    long = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a" * 100},
        ]
    }

    def tokenize(messages, tokenize=False):
        length = 3 if messages[2]["content"] == "a" else 100
        return list(range(length))

    tokenizer.apply_chat_template.side_effect = tokenize

    filtered = filter_by_max_length([short, long], tokenizer, max_length=10)
    assert len(filtered) == 1
    assert filtered[0] is short


def test_stratified_split_preserves_class_ratio() -> None:
    from datasets import Dataset

    rows = []
    for i in range(80):
        rows.append(
            {
                "system": "SYSTEM: Tools available.",
                "chat": (
                    f"USER: News {i}?\n\n"
                    f'ASSISTANT: <functioncall> {{"name": "tool_{i}"}} <|endoftext|>'
                ),
            }
        )
    for i in range(50):
        rows.append(
            {
                "system": "SYSTEM: You are helpful.",
                "chat": (
                    f"USER: Question {i}?\n\n"
                    f"ASSISTANT: Answer {i}. <|endoftext|>"
                ),
            }
        )

    train_ds, eval_ds = preprocess_dataset(Dataset.from_list(rows), max_samples=100)

    def positive_ratio(ds) -> float:
        pos = sum(1 for row in ds if "<functioncall>" in row["messages"][2]["content"])
        return pos / len(ds)

    assert 0.58 <= positive_ratio(train_ds) <= 0.62
    assert 0.58 <= positive_ratio(eval_ds) <= 0.62


def find_sample(dataset, predicate):
    for row in dataset:
        if predicate(row):
            return row
    raise RuntimeError("No matching sample found")


def test_real_samples() -> None:
    print("=== Real Glaive samples ===")

    ds = load_glaive_dataset()

    tool_call_raw = find_sample(ds, lambda r: "<functioncall>" in r["chat"])
    plain_text_raw = find_sample(
        ds,
        lambda r: "<functioncall>" not in r["chat"] and preprocess_sample(r) is not None,
    )

    tool_call_result = preprocess_sample(tool_call_raw)
    plain_text_result = preprocess_sample(plain_text_raw)

    assert_clean_result(tool_call_result, expect_functioncall=True)
    assert_clean_result(plain_text_result, expect_functioncall=False)

    print_result("Tool-call", tool_call_result)
    print_result("Plain-text", plain_text_result)


def main():
    test_raw_regex()
    test_preprocess_edge_cases()
    test_preprocess_dataset_balances_positive_negative()
    test_preprocess_dataset_train_eval_split()
    test_preprocess_dataset_drops_invalid_rows()
    test_real_samples()
    print("All preprocessing tests passed.")


if __name__ == "__main__":
    main()
