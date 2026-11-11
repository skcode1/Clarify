from pathlib import Path
import cv2
import numpy as np
import pytest
from clarify import enhance, ImageNotFoundError, UnsupportedImageError


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
    assert result_path.name == "sample_enhanced.png"

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