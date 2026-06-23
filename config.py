# config.py
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
HF_MODEL_TAG = "10k-curated-7000steps"  # set to None or "" to skip tagging

#inference.py
INFERENCE_MESSAGES = [
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