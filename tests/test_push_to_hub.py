"""Tests for Hugging Face Hub push and branch-based uploads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from config import HF_IGNORE_PATTERNS
from train import push_to_hub


@dataclass
class MockCommitInfo:
    oid: str = "abc123def456"


@pytest.fixture
def mock_api() -> MagicMock:
    api = MagicMock()
    api.upload_folder.return_value = MockCommitInfo()
    return api


def test_push_to_hub_pushes_to_branch_when_tag_provided(mock_api: MagicMock) -> None:
    with (
        patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}),
        patch("train.HfApi", return_value=mock_api),
    ):
        push_to_hub("/tmp/output", "user/repo", tag="700-steps")

    mock_api.create_repo.assert_called_once()
    mock_api.create_branch.assert_called_once_with(
        repo_id="user/repo",
        branch="700-steps",
        token="hf_test_token",
        repo_type="model",
        exist_ok=True,
    )
    mock_api.upload_folder.assert_called_once_with(
        folder_path="/tmp/output",
        repo_id="user/repo",
        token="hf_test_token",
        revision="700-steps",
        commit_message="Add 700-steps model",
        ignore_patterns=HF_IGNORE_PATTERNS,
    )


@pytest.mark.parametrize("tag", [None, ""])
def test_push_to_hub_skips_tag_when_tag_empty(
    mock_api: MagicMock, tag: str | None
) -> None:
    with (
        patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}),
        patch("train.HfApi", return_value=mock_api),
    ):
        push_to_hub("/tmp/output", "user/repo", tag=tag)

    mock_api.create_branch.assert_not_called()
    mock_api.upload_folder.assert_called_once_with(
        folder_path="/tmp/output",
        repo_id="user/repo",
        token="hf_test_token",
        commit_message="Upload fine-tuned model",
        ignore_patterns=HF_IGNORE_PATTERNS,
    )


def test_push_to_hub_skips_when_no_token(mock_api: MagicMock) -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("train.HfApi", return_value=mock_api),
    ):
        push_to_hub("/tmp/output", "user/repo", tag="700-steps")

    mock_api.create_repo.assert_not_called()
    mock_api.create_branch.assert_not_called()
    mock_api.upload_folder.assert_not_called()
