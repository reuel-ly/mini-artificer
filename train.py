"""Fine-tune SmolLM2 with LoRA on Glaive function-calling data."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi
import os

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
    HF_REPO_NAME,
    HF_MODEL_TAG,
)

from chat_template import patch_chat_template
from data_loader import load_glaive_dataset
from inference import run_inference_tests
from preprocess_data import preprocess_dataset

import wandb


def push_to_hub(output_dir: str, repo_name: str, tag: str | None = None) -> None:
    """Push fine-tuned model to HuggingFace Hub and optionally create a version tag."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("No HF_TOKEN found, skipping hub push")
        return

    api = HfApi()
    api.create_repo(
        repo_id=repo_name,
        token=hf_token,
        exist_ok=True,
    )
    commit_info = api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_name,
        token=hf_token,
        commit_message=f"Add {tag} model" if tag else "Upload fine-tuned model",
    )
    print(f"Model pushed to huggingface.co/{repo_name}")

    if tag:
        api.create_tag(
            repo_id=repo_name,
            tag=tag,
            revision=commit_info.oid,
            tag_message=WANDB_RUN_NAME,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Version tagged as: {tag}")
    else:
        print("Tagging skipped (no tag provided)")


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
    tokenizer = patch_chat_template(tokenizer)

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

    # 9. Run inference smoke tests
    print("Running post-training inference tests...")
    run_inference_tests(output_dir=OUTPUT_DIR)

    # 10. Push to HuggingFace Hub
    push_to_hub(
        output_dir=OUTPUT_DIR,
        repo_name=HF_REPO_NAME,
        tag=HF_MODEL_TAG or None,
    )


if __name__ == "__main__":
    main()
