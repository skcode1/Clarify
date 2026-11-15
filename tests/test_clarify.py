from pathlib import Path

import cv2
import numpy as np
import pytest

from clarify import (
    DEFAULT_ARCH,
    DEFAULT_SCALE,
    MODEL_PRESETS,
    ModelArch,
    clear_session_cache,
    enhance,
    health_check,
)
from clarify.exceptions import (
    ConfigurationError,
    ImageNotFoundError,
    ModelNotFoundError,
)

BUNDLED_ARCH: ModelArch = "edsr_baseline"


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Generate a clean test image."""
    img_path = tmp_path / "sample.jpg"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(arr, (16, 16), 8, (255, 0, 0), -1)
    cv2.imwrite(str(img_path), arr)
    return img_path


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------

def test_model_presets_registered():
    """All documented model architectures are registered in MODEL_PRESETS."""
    for arch in ("span", "sr_compact", "realplskr", "edsr_baseline"):
        assert arch in MODEL_PRESETS
        preset = MODEL_PRESETS[arch]
        assert preset.filename.endswith(".onnx")
        assert preset.approx_size_mb > 0
        assert preset.best_for


def test_default_arch_and_scale():
    """Default arch and scale are sensible values."""
    assert DEFAULT_ARCH in MODEL_PRESETS
    assert DEFAULT_SCALE in (2, 3, 4)


# ---------------------------------------------------------------------------
# AI mode: bundled EDSR baseline (default arch)
# ---------------------------------------------------------------------------

def test_enhance_ai_upscales_dimensions(sample_image: Path):
    result_path = enhance(sample_image, mode="ai", scale=2, arch=BUNDLED_ARCH)
    assert result_path.is_file()
    assert result_path.name == "sample_ai.png"

    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (64, 64, 3)


def test_enhance_ai_scale_3(sample_image: Path):
    result_path = enhance(sample_image, mode="ai", scale=3, arch=BUNDLED_ARCH)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    # Bundled EDSR model is a fixed 2x model; all scales produce 2x output
    assert out_bgr.shape == (64, 64, 3)


def test_enhance_ai_scale_4(sample_image: Path):
    result_path = enhance(sample_image, mode="ai", scale=4, arch=BUNDLED_ARCH)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    # Bundled EDSR model is a fixed 2x model; all scales produce 2x output
    assert out_bgr.shape == (64, 64, 3)


def test_enhance_ai_default_arch(sample_image: Path):
    """Default arch (edsr_baseline) works without explicit arch argument."""
    result_path = enhance(sample_image, mode="ai", scale=2)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Classical modes
# ---------------------------------------------------------------------------

def test_enhance_classical(sample_image: Path):
    result_path = enhance(sample_image, mode="classical")
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_enhance_classical_preserves_dimensions(sample_image: Path):
    result_path = enhance(sample_image, mode="classical", scale=4)
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_enhance_denoise(sample_image: Path):
    out = enhance(sample_image, mode="denoise", strength=1.5)
    img = cv2.imread(str(out))
    assert img.shape == (32, 32, 3)


def test_enhance_deblur(sample_image: Path):
    out = enhance(sample_image, mode="deblur", strength=1.2)
    img = cv2.imread(str(out))
    assert img.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_image_raises_error():
    with pytest.raises(ImageNotFoundError):
        enhance("missing_target_file.png")


def test_invalid_scale_raises_error(sample_image: Path):
    with pytest.raises(ConfigurationError, match="Invalid scale"):
        enhance(sample_image, mode="ai", scale=5)


def test_invalid_mode_raises_error(sample_image: Path):
    with pytest.raises(ConfigurationError, match="Invalid mode"):
        enhance(sample_image, mode="invalid")  # type: ignore[arg-type]


def test_invalid_arch_raises_error(sample_image: Path):
    with pytest.raises(ConfigurationError, match="Invalid arch"):
        enhance(sample_image, mode="ai", scale=2, arch="nonexistent")  # type: ignore[arg-type]


def test_custom_model_path_missing_raises(sample_image: Path):
    with pytest.raises(ModelNotFoundError):
        enhance(sample_image, mode="ai", scale=2, arch="edsr_baseline", model_path="/nonexistent/model.onnx")


def test_face_restore_missing_model_raises(sample_image: Path):
    with pytest.raises(ModelNotFoundError):
        enhance(sample_image, mode="face_restore")


# ---------------------------------------------------------------------------
# Output paths & formats
# ---------------------------------------------------------------------------

def test_custom_output_path(sample_image: Path, tmp_path: Path):
    custom_output = tmp_path / "custom" / "output.png"
    result_path = enhance(sample_image, mode="ai", scale=2, arch=BUNDLED_ARCH, output=custom_output)
    assert result_path == custom_output
    assert result_path.is_file()


def test_relative_output_path(sample_image: Path):
    result_path = enhance(sample_image, mode="ai", scale=2, arch=BUNDLED_ARCH, output="relative_output.png")
    assert result_path.name == "relative_output.png"
    assert result_path.is_file()


def test_webp_input_support(tmp_path: Path):
    img_path = tmp_path / "sample.webp"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), arr)
    result_path = enhance(img_path, mode="ai", scale=2, arch=BUNDLED_ARCH)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_png_input_support(tmp_path: Path):
    img_path = tmp_path / "sample.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), arr)
    result_path = enhance(img_path, mode="ai", scale=2, arch=BUNDLED_ARCH)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_output_formats(sample_image: Path):
    for fmt in ["png", "jpeg", "webp", "tiff"]:
        out = enhance(sample_image, mode="classical", output_format=fmt, quality=90)
        assert out.is_file()
        expected = ".jpg" if fmt == "jpeg" else f".{fmt}"
        assert out.suffix.lower() == expected


def test_output_format_jpeg_alias(sample_image: Path):
    out = enhance(sample_image, mode="classical", output_format="jpg", quality=90)
    assert out.is_file()
    assert out.suffix.lower() == ".jpg"


def test_quality_clamped_to_bounds(sample_image: Path):
    out_low = enhance(sample_image, mode="classical", output_format="jpeg", quality=0)
    out_high = enhance(sample_image, mode="classical", output_format="jpeg", quality=200)
    assert out_low.is_file()
    assert out_high.is_file()


def test_unsupported_format_raises(sample_image: Path):
    with pytest.raises(ConfigurationError, match="Unsupported output format"):
        enhance(sample_image, mode="classical", output_format="bmp")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check_structure():
    h = health_check()
    assert "providers_available" in h
    assert "providers_preferred" in h
    assert "bundled_models" in h
    assert "cached_sessions" in h
    assert "default_arch" in h
    assert "default_scale" in h
    assert "version" in h
    assert h["version"] == "0.4.0"
    assert "CPUExecutionProvider" in h["providers_available"]


def test_health_check_bundled_models():
    h = health_check()
    models = h["bundled_models"]
    assert "edsr_baseline" in models
    # edsr_baseline model file should exist
    edsr_entry = models["edsr_baseline"]
    assert edsr_entry[1] is True  # bundled flag


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

def test_clear_session_cache(sample_image: Path):
    """clear_session_cache empties the internal session cache."""
    from clarify.core import _CACHED_SESSIONS

    # Load a session by running inference
    enhance(sample_image, mode="ai", scale=2, arch=BUNDLED_ARCH)
    assert len(_CACHED_SESSIONS) >= 1

    clear_session_cache()
    assert len(_CACHED_SESSIONS) == 0

