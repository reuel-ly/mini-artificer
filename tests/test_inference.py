from unittest.mock import MagicMock, patch

import torch

from config import (
    MAX_NEW_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    REPETITION_PENALTY,
    TEMPERATURE,
    WEATHER_TOOL_SCHEMA,
    format_system_with_tools,
)
from inference import run_inference


def _make_mock_tokenizer(input_ids: torch.Tensor) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = input_ids
    tokenizer.eos_token_id = 0
    tokenizer.decode.return_value = "generated text"
    return tokenizer


def _make_mock_model(input_ids: torch.Tensor, num_new_tokens: int = 3) -> MagicMock:
    model = MagicMock()
    new_tokens = torch.cat(
        [input_ids, torch.arange(1, num_new_tokens + 1).unsqueeze(0)],
        dim=-1,
    )
    model.generate.return_value = new_tokens
    model.parameters.return_value = iter([torch.zeros(1)])
    return model


def test_run_inference_passes_generation_kwargs() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer = _make_mock_tokenizer(input_ids)
    mock_model = _make_mock_model(input_ids)

    run_inference(
        "Hello",
        WEATHER_TOOL_SCHEMA,
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    mock_model.generate.assert_called_once()
    _, kwargs = mock_model.generate.call_args
    assert kwargs["max_new_tokens"] == MAX_NEW_TOKENS
    assert kwargs["temperature"] == TEMPERATURE
    assert kwargs["do_sample"] is True
    assert kwargs["pad_token_id"] == mock_tokenizer.eos_token_id
    assert kwargs["repetition_penalty"] == REPETITION_PENALTY
    assert kwargs["no_repeat_ngram_size"] == NO_REPEAT_NGRAM_SIZE
    assert torch.equal(kwargs["attention_mask"], torch.ones_like(input_ids))


def test_run_inference_builds_messages_from_args() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer = _make_mock_tokenizer(input_ids)
    mock_model = _make_mock_model(input_ids)
    prompt = "What is the weather in Manila?"

    run_inference(
        prompt,
        WEATHER_TOOL_SCHEMA,
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    messages = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == format_system_with_tools(WEATHER_TOOL_SCHEMA)
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == prompt


def test_load_model_and_tokenizer_patches_chat_template() -> None:
    from inference import _load_model_and_tokenizer

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "</s>"

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])

    with (
        patch("inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("inference.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
        patch("inference.PeftModel.from_pretrained", return_value=mock_model),
        patch("inference.patch_chat_template", side_effect=lambda t: t) as mock_patch,
    ):
        _load_model_and_tokenizer("/tmp/output")

    mock_patch.assert_called_once_with(mock_tokenizer)
