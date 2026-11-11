"""Automated tests for the clarify-ai MCP server.

Covers tool registration, argument validation, the upscale_image tool end-to-end,
and the FastMCP in-process call_tool path used by Claude Desktop / Cursor.
"""

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from clarify.mcp_server import mcp, upscale_image


def _run(coro):
    """Run a coroutine on a fresh event loop (pytest-asyncio independent)."""
    return asyncio.run(coro)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Generate a clean 32x32 test image."""
    img_path = tmp_path / "sample.jpg"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(arr, (16, 16), 8, (255, 0, 0), -1)
    cv2.imwrite(str(img_path), arr)
    return img_path


def test_tool_is_registered():
    """The MCP server exposes an upscale_image tool with a documented schema."""
    tools = _run(mcp.list_tools())
    names = [t.name for t in tools]
    assert "upscale_image" in names

    tool = next(t for t in tools if t.name == "upscale_image")
    schema = tool.inputSchema
    assert schema["type"] == "object"
    assert "image_path" in schema["required"]
    assert schema["properties"]["mode"]["enum"] == [
        "ai",
        "classical",
        "denoise",
        "deblur",
        "face_restore",
    ]
    assert "png" in schema["properties"]["output_format"]["enum"]


def test_upscale_image_ai_2x(sample_image: Path):
    out_path = _run(upscale_image(str(sample_image), scale=2, mode="ai", output_format="png"))
    assert Path(out_path).is_file()
    assert Path(out_path).name == "sample_ai.png"
    out_bgr = cv2.imread(str(out_path))
    assert out_bgr.shape == (64, 64, 3)


def test_upscale_image_ai_scale_3(sample_image: Path):
    out_path = _run(upscale_image(str(sample_image), scale=3, mode="ai", output_format="png"))
    out_bgr = cv2.imread(str(out_path))
    assert out_bgr.shape == (96, 96, 3)


def test_upscale_image_ai_scale_4(sample_image: Path):
    out_path = _run(upscale_image(str(sample_image), scale=4, mode="ai", output_format="png"))
    out_bgr = cv2.imread(str(out_path))
    assert out_bgr.shape == (128, 128, 3)


def test_upscale_image_classical_preserves_dimensions(sample_image: Path):
    out_path = _run(upscale_image(str(sample_image), mode="classical", output_format="png"))
    out_bgr = cv2.imread(str(out_path))
    assert out_bgr.shape == (32, 32, 3)


def test_upscale_image_multi_format(sample_image: Path):
    for fmt in ["png", "jpeg", "webp", "tiff"]:
        out_path = _run(
            upscale_image(
                str(sample_image),
                mode="classical",
                output_format=fmt,
            )
        )
        assert Path(out_path).is_file()
        expected = ".jpg" if fmt == "jpeg" else f".{fmt}"
        assert Path(out_path).suffix.lower() == expected


def test_upscale_image_mcp_call_tool_end_to_end(sample_image: Path):
    """Exercise the exact in-process path FastMCP uses for Claude Desktop / Cursor."""
    content_blocks, structured = _run(
        mcp.call_tool(
            "upscale_image",
            {
                "image_path": str(sample_image),
                "scale": 2,
                "mode": "ai",
                "output_format": "png",
            },
        )
    )
    assert isinstance(content_blocks, list) and len(content_blocks) >= 1
    text = " ".join(getattr(block, "text", "") or str(block) for block in content_blocks)
    assert str(sample_image.stem) in text or Path(structured["result"]).is_file()


def test_upscale_image_missing_image_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises((FileNotFoundError, OSError)):
        _run(upscale_image(str(missing)))


def test_upscale_image_empty_image_path_raises():
    with pytest.raises(ValueError):
        _run(upscale_image(""))


def test_upscale_image_invalid_scale_for_ai_raises(sample_image: Path):
    with pytest.raises(ValueError, match="Invalid scale"):
        _run(upscale_image(str(sample_image), scale=5, mode="ai"))


def test_upscale_image_invalid_mode_raises(sample_image: Path):
    with pytest.raises(ValueError, match="Unsupported mode"):
        _run(upscale_image(str(sample_image), mode="invalid"))


def test_upscale_image_invalid_output_format_raises(sample_image: Path):
    with pytest.raises(ValueError, match="Unsupported output format"):
        _run(upscale_image(str(sample_image), output_format="bmp"))


def test_upscale_image_scale_zero_raises(sample_image: Path):
    with pytest.raises(ValueError):
        _run(upscale_image(str(sample_image), scale=0))
