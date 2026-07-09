# config.py
import json

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_NAME = "glaiveai/glaive-function-calling-v2"
OUTPUT_DIR = "./outputs/smol-lora"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_DROPOUT = 0.1

# Dataset
DATASET_SIZE = 30_000
POSITIVE_SAMPLE_RATIO = 0.6
TRAIN_SPLIT_RATIO = 0.9
SEED = 42

# Training
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
MAX_STEPS = 8000
WARMUP_STEPS = 200
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_STEPS = 100

# Logging
WANDB_PROJECT = "mini-artificer"
WANDB_RUN_NAME = "smollm2-135m-lora-r16-30k-60-40-8000steps"

# HuggingFace Hub
HF_REPO_NAME = "reuel-ly/mini-artificer"
HF_MODEL_TAG = "30k-60-40-r16-8000steps"  # Hub branch name; set to None or "" to push to main
HF_IGNORE_PATTERNS = [
    "checkpoint-*",
    "training_args.bin",
    "optimizer.pt",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "trainer_state.json",
]

# Inference generation
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3


def format_system_with_tools(tool_schema: dict) -> str:
    return (
        "You are a helpful assistant with access to the following functions. "
        "Use them if required - call a function only when the request needs one; "
        "otherwise answer in plain text without a function call.\n"
        f"{json.dumps(tool_schema, indent=2)}"
    )


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
INFERENCE_MESSAGES = [
    {
        "role": "system",
        "content": format_system_with_tools(WEATHER_TOOL_SCHEMA),
    },
    {
        "role": "user",
        "content": "What is the weather in Manila?",
    },
]
