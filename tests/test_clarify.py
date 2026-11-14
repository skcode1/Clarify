from pathlib import Path

import cv2
import numpy as np
import pytest

from clarify import (
    ConfigurationError,
    ImageNotFoundError,
    ModelNotFoundError,
    enhance,
)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Generate a clean test image."""
    img_path = tmp_path / "sample.jpg"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(arr, (16, 16), 8, (255, 0, 0), -1)
    cv2.imwrite(str(img_path), arr)
    return img_path


def test_enhance_ai_upscales_dimensions(sample_image: Path):
    result_path = enhance(sample_image, mode="ai", scale=2)
    assert result_path.is_file()
    assert result_path.name == "sample_ai.png"

    # Verify 2x upscale dimensions (32x32 -> 64x64)
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (64, 64, 3)


def test_enhance_ai_scale_3(sample_image: Path):
    """Test 3x AI upscaling."""
    result_path = enhance(sample_image, mode="ai", scale=3)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (96, 96, 3)  # 32 * 3 = 96


def test_enhance_ai_scale_4(sample_image: Path):
    """Test 4x AI upscaling."""
    result_path = enhance(sample_image, mode="ai", scale=4)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (128, 128, 3)  # 32 * 4 = 128


def test_enhance_classical(sample_image: Path):
    result_path = enhance(sample_image, mode="classical")
    assert result_path.is_file()

    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_enhance_classical_preserves_dimensions(sample_image: Path):
    """Classical mode should not change dimensions."""
    result_path = enhance(sample_image, mode="classical", scale=4)  # scale ignored
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_missing_image_raises_error():
    with pytest.raises(ImageNotFoundError):
        enhance("missing_target_file.png")


def test_invalid_scale_raises_error(sample_image: Path):
    with pytest.raises(ValueError, match="Invalid scale"):
        enhance(sample_image, mode="ai", scale=5)


def test_invalid_mode_raises_error(sample_image: Path):
    with pytest.raises(ValueError, match="Invalid mode"):
        enhance(sample_image, mode="invalid")


def test_custom_output_path(sample_image: Path, tmp_path: Path):
    """Test custom output path."""
    custom_output = tmp_path / "custom" / "output.png"
    result_path = enhance(sample_image, mode="ai", scale=2, output=custom_output)
    assert result_path == custom_output
    assert result_path.is_file()


def test_relative_output_path(sample_image: Path):
    """Test relative output path resolves correctly."""
    result_path = enhance(sample_image, mode="ai", scale=2, output="relative_output.png")
    assert result_path.name == "relative_output.png"
    assert result_path.is_file()


def test_webp_input_support(tmp_path: Path):
    """Test WebP input format support."""
    img_path = tmp_path / "sample.webp"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), arr)
    result_path = enhance(img_path, mode="ai", scale=2)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_png_input_support(tmp_path: Path):
    """Test PNG input format support."""
    img_path = tmp_path / "sample.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), arr)
    result_path = enhance(img_path, mode="ai", scale=2)
    assert result_path.is_file()
    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_output_formats(sample_image: Path):
    """Multi-format serializer produces correct suffixes for all formats."""
    for fmt in ["png", "jpeg", "webp", "tiff"]:
        out = enhance(sample_image, mode="classical", output_format=fmt, quality=90)
        assert out.is_file()
        expected = ".jpg" if fmt == "jpeg" else f".{fmt}"
        assert out.suffix.lower() == expected


def test_output_format_jpeg_alias(sample_image: Path):
    """'jpg' is accepted as an alias for 'jpeg'."""
    out = enhance(sample_image, mode="classical", output_format="jpg", quality=90)
    assert out.is_file()
    assert out.suffix.lower() == ".jpg"


def test_quality_clamped_to_bounds(sample_image: Path):
    """Quality values outside 1-100 are clamped rather than crashing."""
    out_low = enhance(sample_image, mode="classical", output_format="jpeg", quality=0)
    out_high = enhance(sample_image, mode="classical", output_format="jpeg", quality=200)
    assert out_low.is_file()
    assert out_high.is_file()


def test_unsupported_format_raises(sample_image: Path):
    """An unsupported output format raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Unsupported output format"):
        enhance(sample_image, mode="classical", output_format="bmp")


def test_denoise_preserves_dimensions(sample_image: Path):
    out = enhance(sample_image, mode="denoise", strength=1.5)
    img = cv2.imread(str(out))
    assert img.shape == (32, 32, 3)


def test_deblur_preserves_dimensions(sample_image: Path):
    out = enhance(sample_image, mode="deblur", strength=1.2)
    img = cv2.imread(str(out))
    assert img.shape == (32, 32, 3)


def test_denoise_strength_zero(sample_image: Path):
    """Zero strength should still run and preserve dimensions."""
    out = enhance(sample_image, mode="denoise", strength=0.0)
    img = cv2.imread(str(out))
    assert img.shape == (32, 32, 3)


def test_face_restore_missing_model_raises(sample_image: Path):
    """Face restore without the bundled ONNX artifact raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError):
        enhance(sample_image, mode="face_restore")


def test_classical_preserves_dimensions(sample_image: Path):
    out = enhance(sample_image, mode="classical")
    img = cv2.imread(str(out))
    assert img.shape == (32, 32, 3)