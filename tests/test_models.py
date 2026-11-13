"""Automated tests for EDSR Full (hq) model support."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from clarify import (
    get_edsr_model_path,
    list_available_models,
)
from clarify.core import enhance
from clarify.exceptions import ModelNotFoundError
from clarify.models import FULL_MODEL_REGISTRY


def test_get_edsr_model_path_fast_returns_bundled():
    """precision='fast' should resolve to a bundled baseline model."""
    for scale in (2, 3, 4):
        path = get_edsr_model_path(scale=scale, precision="fast")
        assert path.is_file()
        assert path.name == f"edsr_x{scale}.onnx"
        assert "models" in str(path)


def test_get_edsr_model_path_invalid_scale():
    """Invalid scale should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported scale"):
        get_edsr_model_path(scale=5, precision="fast")


def test_get_edsr_model_path_invalid_precision():
    """Invalid precision should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid precision"):
        get_edsr_model_path(scale=2, precision="ultra")  # type: ignore[arg-type]


def test_get_edsr_model_path_custom_model_exists(tmp_path: Path):
    """Explicit model_path should be returned when the file exists."""
    custom = tmp_path / "custom.onnx"
    custom.write_bytes(b"\x00" * 100)
    path = get_edsr_model_path(scale=2, precision="hq", model_path=str(custom))
    assert path == custom.resolve()


def test_get_edsr_model_path_custom_model_missing():
    """Explicit model_path that doesn't exist should raise ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError, match="Custom model not found"):
        get_edsr_model_path(scale=2, precision="hq", model_path="/nonexistent/model.onnx")


def test_list_available_models_structure():
    """list_available_models should return a dict with x2, x3, x4 keys."""
    summary = list_available_models()
    assert set(summary.keys()) == {"x2", "x3", "x4"}
    for scale_key in ("x2", "x3", "x4"):
        entry = summary[scale_key]
        assert "fast" in entry
        assert "hq" in entry
        assert entry["fast"]["source"] == "bundled"
        assert entry["hq"]["source"] == "cache_or_download"
        assert entry["fast"]["available"] is True


@patch("clarify.models.urllib.request.urlretrieve")
def test_get_edsr_model_path_hq_download_and_cache(mock_urlretrieve, tmp_path, monkeypatch):
    """precision='hq' should download, verify SHA-256, and cache the model."""
    monkeypatch.setattr("clarify.models.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("clarify.models.FULL_MODEL_REGISTRY", {
        2: {
            "filename": "edsr_x2_full.onnx",
            "url": "https://example.com/edsr_x2_full.onnx",
            "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X2",
        },
        3: FULL_MODEL_REGISTRY[3],
        4: FULL_MODEL_REGISTRY[4],
    })

    # Simulate a downloaded file by having urlretrieve create it
    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"\x00" * 1000)

    mock_urlretrieve.side_effect = fake_urlretrieve

    path = get_edsr_model_path(scale=2, precision="hq")
    assert path.is_file()
    assert path.name == "edsr_x2_full.onnx"
    mock_urlretrieve.assert_called_once()


@patch("clarify.models.urllib.request.urlretrieve")
def test_get_edsr_model_path_hq_sha256_mismatch(mock_urlretrieve, tmp_path, monkeypatch):
    """SHA-256 mismatch on downloaded model should raise ModelNotFoundError."""
    monkeypatch.setattr("clarify.models.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("clarify.models.FULL_MODEL_REGISTRY", {
        2: {
            "filename": "edsr_x2_full.onnx",
            "url": "https://example.com/edsr_x2_full.onnx",
            "sha256": "deadbeef" * 8,  # invalid hash
        },
        3: FULL_MODEL_REGISTRY[3],
        4: FULL_MODEL_REGISTRY[4],
    })

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"\x00" * 1000)

    mock_urlretrieve.side_effect = fake_urlretrieve

    with pytest.raises(ModelNotFoundError, match="SHA-256 integrity check"):
        get_edsr_model_path(scale=2, precision="hq")


def test_enhance_ai_fast_precision(sample_image: Path, tmp_path: Path):
    """enhance with precision='fast' should work using bundled baseline model."""
    result = enhance(sample_image, mode="ai", scale=2, precision="fast")
    assert result.is_file()
    out_bgr = cv2.imread(str(result))
    assert out_bgr.shape == (64, 64, 3)  # 2x upscale


def test_enhance_ai_fast_precision_scale_4(sample_image: Path):
    """enhance with precision='fast' scale=4 should produce 4x dimensions."""
    result = enhance(sample_image, mode="ai", scale=4, precision="fast")
    assert result.is_file()
    out_bgr = cv2.imread(str(result))
    assert out_bgr.shape == (128, 128, 3)  # 4x upscale


# --- shared fixture ---

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Generate a clean 32x32 test image."""
    img_path = tmp_path / "sample.jpg"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(arr, (16, 16), 8, (255, 0, 0), -1)
    cv2.imwrite(str(img_path), arr)
    return img_path
