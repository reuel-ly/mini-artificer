"""Fine-tune SmolLM2 with LoRA on Glaive function-calling data."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi
import os

from config import (
    BATCH_SIZE,
    DATASET_SIZE,
    EVAL_STEPS,
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
    TRAIN_SPLIT_RATIO,
    SEED,
    WARMUP_STEPS,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    HF_REPO_NAME,
    HF_MODEL_TAG,
    HF_IGNORE_PATTERNS,
)

from chat_template import patch_chat_template
from data_loader import load_glaive_dataset
from inference import run_inference_tests
from preprocess_data import preprocess_dataset

import wandb

_ACCELERATE_LAUNCH_HINT = (
    "Launch multi-GPU training with DDP via Accelerate, e.g.\n"
    "  accelerate launch --multi_gpu --mixed_precision fp16 train.py\n"
    "or\n"
    "  accelerate launch --config_file accelerate_config.yaml train.py"
)


def _is_main_process() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in (-1, 0)


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or os.environ.get("LOCAL_RANK") is not None


def _require_ddp_for_multi_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return
    if _is_distributed_launch():
        return
    raise RuntimeError(
        "Multiple GPUs detected, but training was started with plain `python train.py`. "
        "HuggingFace Trainer falls back to DataParallel in that mode, which breaks PEFT/LoRA.\n"
        f"{_ACCELERATE_LAUNCH_HINT}"
    )


def push_to_hub(output_dir: str, repo_name: str, tag: str | None = None) -> None:
    """Push fine-tuned model to HuggingFace Hub on a named branch (tag) or main."""
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

    if tag:
        api.create_branch(
            repo_id=repo_name,
            branch=tag,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_name,
            token=hf_token,
            revision=tag,
            commit_message=f"Add {tag} model",
            ignore_patterns=HF_IGNORE_PATTERNS,
        )
        print(f"Pushed to branch '{tag}' on huggingface.co/{repo_name}")
    else:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_name,
            token=hf_token,
            commit_message="Upload fine-tuned model",
            ignore_patterns=HF_IGNORE_PATTERNS,
        )
        print(f"Model pushed to huggingface.co/{repo_name}")


def main() -> None:
    _require_ddp_for_multi_gpu()
    set_seed(SEED)

    try:
        # 1. Load the dataset
        raw_ds = load_glaive_dataset()

        # 2. Load tokenizer (needed for length filtering during preprocessing)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer = patch_chat_template(tokenizer)

        # 3. Preprocess and curate train/eval splits
        train_ds, eval_ds = preprocess_dataset(
            raw_ds,
            max_samples=DATASET_SIZE,
            train_split_ratio=TRAIN_SPLIT_RATIO,
            tokenizer=tokenizer,
            max_length=MAX_SEQ_LENGTH,
        )
        print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
        print(f"Sample 0: {train_ds[0]}")

        # 4. Load SmolLM2-135M-Instruct
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # 5. Apply LoRA config
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

        # 6. Define training arguments
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
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=EVAL_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if _is_main_process() else "none",
            run_name=WANDB_RUN_NAME,
            ddp_find_unused_parameters=False,
            seed=SEED,
        )

        # 7. Initialize SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

        if _is_main_process():
            wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

        # 8. Train
        trainer.train()

        if trainer.is_world_process_zero():
            # 9. Save the model
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Model saved to {OUTPUT_DIR}")

            # 10. Run inference smoke tests
            print("Running post-training inference tests...")
            run_inference_tests(output_dir=OUTPUT_DIR)

            # 11. Push to HuggingFace Hub
            push_to_hub(
                output_dir=OUTPUT_DIR,
                repo_name=HF_REPO_NAME,
                tag=HF_MODEL_TAG or None,
            )
    finally:
        if _is_main_process() and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
