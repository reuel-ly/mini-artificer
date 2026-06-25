"""Tests for Glaive dataset loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_loader import load_glaive_dataset


def test_load_glaive_dataset_propagates_non_import_errors() -> None:
    with patch("datasets.load_dataset", side_effect=ConnectionError("network down")):
        with pytest.raises(ConnectionError, match="network down"):
            load_glaive_dataset()


def test_load_glaive_dataset_falls_back_on_import_error() -> None:
    mock_dataset = MagicMock()
    with (
        patch("datasets.load_dataset", side_effect=ImportError("no datasets")),
        patch("data_loader._load_from_json", return_value=mock_dataset) as mock_json,
    ):
        result = load_glaive_dataset()
    mock_json.assert_called_once()
    assert result is mock_dataset
