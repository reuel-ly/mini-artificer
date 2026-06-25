"""Test the fine-tuned model on tool calling prompts."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from chat_template import patch_chat_template
from config import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    NO_REPEAT_NGRAM_SIZE,
    OUTPUT_DIR,
    REPETITION_PENALTY,
    TEMPERATURE,
    WEATHER_TOOL_SCHEMA,
    format_system_with_tools,
)

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
            "content": format_system_with_tools(tool_schema),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    input_ids = (
        encoded.to(device)
        if isinstance(encoded, torch.Tensor)
        else encoded["input_ids"].to(device)
    )
    attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    )

    prompt_length = input_ids.shape[-1]
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

    debug_messages = [
        {
            "role": "system",
            "content": format_system_with_tools(WEATHER_TOOL_SCHEMA),
        },
        {
            "role": "user",
            "content": "What is the weather in Manila?",
        },
    ]
    debug_inputs = tokenizer.apply_chat_template(
        debug_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    # Debug 1 — see exactly what goes into the model
    print("=== RAW TOKENIZED INPUT ===")
    debug_ids = (
        debug_inputs
        if isinstance(debug_inputs, torch.Tensor)
        else debug_inputs["input_ids"]
    )
    print(tokenizer.decode(debug_ids[0]))
    print("===========================")

    # Debug 2 — check if chat template has generation markers
    print("=== CHAT TEMPLATE ===")
    print(tokenizer.chat_template)
    print("=====================")


if __name__ == "__main__":
    run_inference_tests()
