# mini-artificer

LoRA fine-tuning of SmolLM2-135M-Instruct on Glaive function-calling data for tool use.

## Models & Data

| Resource | Link | Note |
|----------|------|------|
| Fine-tuned model | [reuel-ly/mini-artificer](https://huggingface.co/reuel-ly/mini-artificer) | LoRA adapter; best run (`10k-curated-7000steps`) |
| Base model | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | Model family; training uses [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| Dataset | [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Function-calling SFT data |

## Overview

This project teaches a small language model to emit structured tool/function calls from natural-language prompts. It fine-tunes [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) with LoRA (r=4) using TRL's `SFTTrainer` and assistant-only loss. Training data is a curated 10k-sample subset of [Glaive function-calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2), preprocessed into chat format in `preprocess_data.py`. The resulting adapter is pushed to the Hugging Face Hub from `train.py`.

## Training Approach

- LoRA: r=4, alpha=8, targets `q_proj` / `k_proj` / `v_proj`
- 10k curated samples, max 512 tokens, 7000 steps (published run)
- Learning rate 5e-4, batch 4 × grad accum 4
- W&B project: `mini-artificer`

## Training Results

Runs 1–2 used exploratory training on larger raw subsets (112k samples, partial epochs). Runs 3–4 switched to a curated 10k subset; **Run 4** is the production checkpoint published to the Hub (best loss 0.1461 @ step 4890, best token accuracy 97.99%).

| Run | Steps | Dataset | Final Loss | Best Loss | Final Token Accuracy | Best Token Accuracy |
|-----|-------|---------|------------|-----------|----------------|---------------------|
| Baseline | 350 | 112k (0.1 epoch) | 0.8314 | Step 140: 0.7482 | 79.5% | Step 280: 82.01% |
| Run 2 | 700 | 112k (0.1 epoch) | 0.7679 | Step 500: 0.7186 | 81.89% | Step 500: 83.03% |
| Run 3 | 700 | 10k curated (1.1 epoch) | 0.4181 | 420 Steps: 0.2927 | 91.59% | 420 Steps: 93.91% |
| Run 4 | 7000 | 10k curated | 0.3408 | 4890 Steps: 0.1461 | 92.59% | 4890 Steps: 97.99% |

## Repository

- `train.py` — fine-tune and push to Hub
- `inference.py` — smoke-test tool-calling prompts
- `config.py` — hyperparameters and Hub repo name
