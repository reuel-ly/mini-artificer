"""Chat template helpers for SmolLM2 + TRL training."""


def patch_chat_template(tokenizer):
    """Patch SmolLM2 chat template to be TRL training-compatible.

    ``<|im_end|>`` must sit inside ``{% generation %}`` so ``assistant_only_loss``
    supervises the turn-ending stop token.
    """
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{% generation %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endgeneration %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )
    return tokenizer
