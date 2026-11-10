from pathlib import Path
import cv2
import numpy as np
import pytest
from clarify import enhance, ImageNotFoundError


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


def test_enhance_classical(sample_image: Path):
    result_path = enhance(sample_image, mode="classical")
    assert result_path.is_file()

    out_bgr = cv2.imread(str(result_path))
    assert out_bgr.shape == (32, 32, 3)


def test_missing_image_raises_error():
    with pytest.raises(ImageNotFoundError):
        enhance("missing_target_file.png")