"""Tests for multi-GPU launch guards in train.py."""

from __future__ import annotations

import pytest

from train import _require_ddp_for_multi_gpu


def test_require_ddp_allows_single_gpu_without_distributed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("train.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("train.torch.cuda.device_count", lambda: 1)

    _require_ddp_for_multi_gpu()


def test_require_ddp_allows_multi_gpu_with_accelerate_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr("train.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("train.torch.cuda.device_count", lambda: 2)

    _require_ddp_for_multi_gpu()


def test_require_ddp_rejects_plain_python_on_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("train.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("train.torch.cuda.device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="DataParallel"):
        _require_ddp_for_multi_gpu()
