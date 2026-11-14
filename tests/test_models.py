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
    """Explicit model_path should be returned when the file exists and is valid ONNX."""
    # Use a real bundled baseline model as a valid custom model
    custom = Path("src/clarify/models/edsr_x2.onnx")
    path = get_edsr_model_path(scale=2, precision="hq", model_path=str(custom))
    assert path == custom.resolve()


def test_get_edsr_model_path_custom_model_missing():
    """Explicit model_path that doesn't exist should raise ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError, match="Custom model not found"):
        get_edsr_model_path(scale=2, precision="hq", model_path="/nonexistent/model.onnx")


def test_get_edsr_model_path_custom_model_invalid_onnx(tmp_path: Path):
    """Explicit model_path that is not a valid ONNX model should raise ModelNotFoundError."""
    invalid = tmp_path / "not_onnx.onnx"
    invalid.write_bytes(b"this is not an onnx file")
    with pytest.raises(ModelNotFoundError, match="Failed to load ONNX model"):
        get_edsr_model_path(scale=2, precision="hq", model_path=str(invalid))


def test_placeholder_hash_rejected():
    """Placeholder SHA-256 hashes should be rejected with a clear error message."""
    # Temporarily replace the registry with placeholder hashes
    original_registry = dict(FULL_MODEL_REGISTRY)
    try:
        FULL_MODEL_REGISTRY[2] = {
            "filename": "edsr_x2_full.onnx",
            "url": "https://example.com/edsr_x2_full.onnx",
            "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X2",
        }
        with pytest.raises(ModelNotFoundError, match="placeholder"):
            get_edsr_model_path(scale=2, precision="hq")
    finally:
        FULL_MODEL_REGISTRY.clear()
        FULL_MODEL_REGISTRY.update(original_registry)


@patch("clarify.models._download_file")
def test_get_edsr_model_path_hq_download_and_cache(mock_download, tmp_path, monkeypatch):
    """precision='hq' should download, verify SHA-256, and cache the model."""
    monkeypatch.setattr("clarify.models.CACHE_DIR", tmp_path / "cache")
    # Use a real SHA-256 hash of a dummy file
    dummy_content = b"\x00" * 1000
    import hashlib
    real_hash = hashlib.sha256(dummy_content).hexdigest()

    monkeypatch.setattr("clarify.models.FULL_MODEL_REGISTRY", {
        2: {
            "filename": "edsr_x2_full.onnx",
            "url": "https://example.com/edsr_x2_full.onnx",
            "sha256": real_hash,
        },
        3: FULL_MODEL_REGISTRY[3],
        4: FULL_MODEL_REGISTRY[4],
    })

    # Simulate a downloaded file by having _download_file create it
    def fake_download(url, filename, timeout=300):
        Path(filename).write_bytes(dummy_content)

    mock_download.side_effect = fake_download

    path = get_edsr_model_path(scale=2, precision="hq")
    assert path.is_file()
    assert path.name == "edsr_x2_full.onnx"
    mock_download.assert_called_once()


@patch("clarify.models._download_file")
def test_get_edsr_model_path_hq_sha256_mismatch(mock_download, tmp_path, monkeypatch):
    """SHA-256 mismatch on downloaded model should raise ModelNotFoundError."""
    monkeypatch.setattr("clarify.models.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("clarify.models.FULL_MODEL_REGISTRY", {
        2: {
            "filename": "edsr_x2_full.onnx",
            "url": "https://example.com/edsr_x2_full.onnx",
            "sha256": "deadbeef" * 8,  # invalid hash that won't match
        },
        3: FULL_MODEL_REGISTRY[3],
        4: FULL_MODEL_REGISTRY[4],
    })

    def fake_download(url, filename, timeout=300):
        Path(filename).write_bytes(b"\x00" * 1000)

    mock_download.side_effect = fake_download

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


def test_list_available_models_structure():
    """list_available_models should return a dict with x2, x3, x4, face_restore keys."""
    summary = list_available_models()
    assert set(summary.keys()) == {"x2", "x3", "x4", "face_restore"}
    for scale_key in ("x2", "x3", "x4"):
        entry = summary[scale_key]
        assert "fast" in entry
        assert "hq" in entry
        assert entry["fast"]["source"] == "bundled"
        assert entry["fast"]["available"] is True
        assert "hash_configured" in entry["hq"]
    # face_restore entry
    fr_entry = summary["face_restore"]
    assert "gfpgan_v1_4" in fr_entry
    assert fr_entry["gfpgan_v1_4"]["source"] in ("bundled", "cache_or_download")
    assert "hash_configured" in fr_entry["gfpgan_v1_4"]


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Generate a clean 32x32 test image."""
    img_path = tmp_path / "sample.jpg"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(arr, (16, 16), 8, (255, 0, 0), -1)
    cv2.imwrite(str(img_path), arr)
    return img_path
