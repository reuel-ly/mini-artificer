# config.py
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_NAME = "glaiveai/glaive-function-calling-v2"
OUTPUT_DIR = "./outputs/smol-lora"
LORA_R = 4
LORA_ALPHA = 8
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-4
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
MAX_STEPS = 350