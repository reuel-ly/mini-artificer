from data_loader import load_glaive_dataset
from preprocess_data import ASSISTANT_RE, USER_RE, preprocess_dataset, preprocess_sample


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

    positive_chat = (
        "USER: US news?\n\n"
        'ASSISTANT: <functioncall> {"name": "get_news"} <|endoftext|>'
    )
    negative_chat = (
        "USER: Can you book a flight?\n\n"
        "ASSISTANT: I'm sorry, I can't. <|endoftext|>"
    )

    rows = []
    for i in range(10):
        rows.append({"system": "SYSTEM: Tools available.", "chat": positive_chat})
        rows.append({"system": "SYSTEM: You are helpful.", "chat": negative_chat})

    result = preprocess_dataset(Dataset.from_list(rows), max_samples=10)

    assert len(result) == 10
    positive_count = sum(
        1 for row in result if "<functioncall>" in row["messages"][2]["content"]
    )
    negative_count = len(result) - positive_count
    assert positive_count == 8
    assert negative_count == 2


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

    result = preprocess_dataset(ds)

    assert len(result) == 2
    for row in result:
        assert row["messages"] is not None
        assert len(row["messages"]) == 3
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]


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
    test_preprocess_dataset_drops_invalid_rows()
    test_real_samples()
    print("All preprocessing tests passed.")


if __name__ == "__main__":
    main()
