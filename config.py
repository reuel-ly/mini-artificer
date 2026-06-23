# config.py
import json

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_NAME = "glaiveai/glaive-function-calling-v2"
OUTPUT_DIR = "./outputs/smol-lora"

# LoRA
LORA_R = 4
LORA_ALPHA = 8
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]
LORA_DROPOUT = 0.05

# Dataset
DATASET_SIZE = 10_000

# Training
LEARNING_RATE = 5e-4
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
MAX_STEPS = 7000
WARMUP_STEPS = 200
GRADIENT_ACCUMULATION_STEPS = 4

# Logging
WANDB_PROJECT = "mini-artificer"
WANDB_RUN_NAME = "smollm2-135m-lora-r4-10k-curated-7000steps"

# HuggingFace Hub
HF_REPO_NAME = "reuel-ly/mini-artificer"
HF_MODEL_TAG = "10k-curated-7000steps"  # Hub branch name; set to None or "" to push to main
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
NO_REPEAT_NGRAM_SIZE = 10


def format_system_with_tools(tool_schema: dict) -> str:
    return (
        "You are a helpful assistant with access to the following functions. "
        f"Use them if required -\n{json.dumps(tool_schema, indent=2)}"
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