from transformers import AutoTokenizer
from trl.chat_template_utils import has_generation_markers

from chat_template import patch_chat_template
from config import MODEL_NAME

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


def test_patch_chat_template_adds_generation_markers() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    original_template = tokenizer.chat_template

    patch_chat_template(tokenizer)

    assert has_generation_markers(tokenizer.chat_template)
    assert (
        "{% generation %}{{ message['content'] }}<|im_end|>\n{% endgeneration %}"
        in tokenizer.chat_template
    )

    original_text = tokenizer.apply_chat_template(
        SAMPLE_MESSAGES, tokenize=False, chat_template=original_template
    )
    patched_text = tokenizer.apply_chat_template(SAMPLE_MESSAGES, tokenize=False)
    assert patched_text == original_text


if __name__ == "__main__":
    test_patch_chat_template_adds_generation_markers()
    print("Chat template test passed.")
