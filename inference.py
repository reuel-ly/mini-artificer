"""Test the fine-tuned model on tool calling prompts."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

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


class StopAfterFunctionCall(StoppingCriteria):
    """Stop once a complete function-call block has been generated."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_length :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if "</functioncall>" in text:
            return True
        if "<functioncall>" not in text:
            return False
        after = text.split("<functioncall>", 1)[1].strip()
        return (
            after.startswith("{")
            and after.count("{") == after.count("}")
            and after.endswith("}")
        )


def _generation_eos_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Stop generation on standard EOS and SmolLM2 chat turn markers."""
    eos_ids = {tokenizer.eos_token_id}
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(im_end_id)
    return sorted(eos_ids)


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
    prompt_length = input_ids.shape[-1]

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=_generation_eos_token_ids(tokenizer),
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        stopping_criteria=StoppingCriteriaList(
            [StopAfterFunctionCall(tokenizer, prompt_length)]
        ),
    )

    new_tokens = output[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_inference_tests(output_dir: str = OUTPUT_DIR) -> None:
    model, tokenizer = _load_model_and_tokenizer(output_dir)

    # --- Positive: direct weather query ---
    print("=== Test 1: Should call tool (weather query) ===")
    result = run_inference(
        "What is the weather in Manila?",
        WEATHER_TOOL_SCHEMA,
        model=model,
        tokenizer=tokenizer,
    )
    print(result)
    print('='*10)

    # --- Positive: different city, confirms argument extraction ---
    print("\n=== Test 2: Should call tool (different city) ===")
    result = run_inference(
        "How is the weather in Tokyo today?",
        WEATHER_TOOL_SCHEMA,
        model=model,
        tokenizer=tokenizer,
    )
    print(result)
    print('='*10)

    # --- Easy negative: completely off-topic (no location, no weather) ---
    print("\n=== Test 3: Should NOT call tool (easy — unrelated query) ===")
    result = run_inference(
        "Can you tell me a joke?",
        WEATHER_TOOL_SCHEMA,
        model=model,
        tokenizer=tokenizer,
    )
    print(result)
    print('='*10)

    # --- Hard negative: information lookup, not weather ---
    print("\n=== Test 4: Should NOT call tool (hard — info query, wrong domain) ===")
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
