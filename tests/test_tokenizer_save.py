"""Verify patched chat template survives tokenizer save/load."""

from __future__ import annotations

import tempfile
from pathlib import Path

from transformers import AutoTokenizer
from trl.chat_template_utils import has_generation_markers

from chat_template import patch_chat_template
from config import MODEL_NAME


def _load_saved_chat_template(output_dir: Path) -> str:
    """Load chat template from saved tokenizer artifacts."""
    jinja_path = output_dir / "chat_template.jinja"
    if jinja_path.exists():
        return jinja_path.read_text(encoding="utf-8")

    config_path = output_dir / "tokenizer_config.json"
    if config_path.exists():
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if template:
            return template

    raise FileNotFoundError(f"No chat template found in {output_dir}")


def test_patched_chat_template_survives_save_pretrained() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    patch_chat_template(tokenizer)
    assert has_generation_markers(tokenizer.chat_template)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        tokenizer.save_pretrained(output_dir)

        saved_template = _load_saved_chat_template(output_dir)
        assert has_generation_markers(saved_template), (
            "Saved chat template is missing TRL generation markers. "
            f"Checked files in {list(output_dir.iterdir())}"
        )

        jinja_path = output_dir / "chat_template.jinja"
        if jinja_path.exists():
            assert has_generation_markers(jinja_path.read_text(encoding="utf-8")), (
                "chat_template.jinja is missing TRL generation markers"
            )

        reloaded = AutoTokenizer.from_pretrained(output_dir)
        assert has_generation_markers(reloaded.chat_template), (
            "Reloaded tokenizer.chat_template is missing TRL generation markers"
        )


if __name__ == "__main__":
    test_patched_chat_template_survives_save_pretrained()
    print("Tokenizer save test passed.")
