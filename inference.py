"""Test the fine-tuned model on tool calling prompts."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from chat_template import patch_chat_template
from config import MODEL_NAME, OUTPUT_DIR

WEATHER_TOOL_SCHEMA = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
}


def _load_model_and_tokenizer(output_dir: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = patch_chat_template(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(model, output_dir)
    model.eval()
    model.to(device)
    return model, tokenizer


def run_inference(
    prompt: str,
    tool_schema: dict,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    output_dir: str = OUTPUT_DIR,
) -> str:
    if model is None or tokenizer is None:
        model, tokenizer = _load_model_and_tokenizer(output_dir)

    device = next(model.parameters()).device

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to the following functions. "
                f"Use them if required -\n{tool_schema}"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True,
    )

    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = output[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_inference_tests(output_dir: str = OUTPUT_DIR) -> None:
    model, tokenizer = _load_model_and_tokenizer(output_dir)

    print("=== Test 1: Should call tool ===")
    result = run_inference(
        "What is the weather in Manila?",
        WEATHER_TOOL_SCHEMA,
        model=model,
        tokenizer=tokenizer,
    )
    print(result)
    
    print('='*10)

    print("\n=== Test 2: Should NOT call tool ===")
    result = run_inference(
        "What is the capital of the Philippines?",
        WEATHER_TOOL_SCHEMA,
        model=model,
        tokenizer=tokenizer,
    )
    print(result)

    print('='*10)
    
    # Debug 1 — see exactly what goes into the model
    print("=== RAW TOKENIZED INPUT ===")
    print(tokenizer.decode(input_ids[0]))
    print("===========================")

    # Debug 2 — check if chat template has generation markers
    print("=== CHAT TEMPLATE ===")
    print(tokenizer.chat_template)
    print("=====================")


if __name__ == "__main__":
    run_inference_tests()
