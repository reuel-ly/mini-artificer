"""Fine-tune SmolLM2 with LoRA on Glaive function-calling data."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


from config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    MODEL_NAME,
    OUTPUT_DIR,
    WARMUP_STEPS,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)

from data_loader import load_glaive_dataset
from preprocess_data import preprocess_dataset

import wandb




def main() -> None:
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    # 1. Load the dataset
    raw_ds = load_glaive_dataset()

    # 2. Preprocess using preprocess_sample (via preprocess_dataset)
    train_ds = preprocess_dataset(raw_ds)
    print(f"Preprocessed {len(train_ds)} samples")
    print(f"Sample 0: {train_ds[0]}")  

    # 3. Load SmolLM2-135M-Instruct
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Apply LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Define training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_length=MAX_SEQ_LENGTH,
        assistant_only_loss=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=MAX_STEPS,
        report_to="wandb",
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # 7. Train
    trainer.train()

    # 8. Save the model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
