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

BUNDLED_ARCH = "edsr_baseline"


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


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_tools_registered():
    """The MCP server exposes all documented tools."""
    tools = _run(mcp.list_tools())
    names = [t.name for t in tools]
    for expected in ("upscale_image", "health", "list_models", "batch_upscale"):
        assert expected in names, f"Tool '{expected}' not registered"


def test_upscale_image_tool_schema():
    """upscale_image tool has the documented input schema."""
    tools = _run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "upscale_image")
    schema = tool.inputSchema
    assert schema["type"] == "object"
    assert "image_path" in schema["required"]
    assert "arch" in schema["properties"]
    assert schema["properties"]["arch"]["enum"] == [
        "span",
        "sr_compact",
        "realplskr",
        "edsr_baseline",
    ]
    assert schema["properties"]["mode"]["enum"] == [
        "ai",
        "classical",
        "denoise",
        "deblur",
        "face_restore",
    ]


# ---------------------------------------------------------------------------
# upscale_image
# ---------------------------------------------------------------------------

def test_upscale_image_ai_2x(sample_image: Path):
    result = _run(upscale_image(str(sample_image), scale=2, mode="ai", arch=BUNDLED_ARCH, output_format="png"))
    assert Path(result["output_path"]).is_file()
    assert Path(result["output_path"]).name == "sample_ai.png"
    out_bgr = cv2.imread(str(result["output_path"]))
    assert out_bgr.shape == (64, 64, 3)


def test_upscale_image_ai_scale_3(sample_image: Path):
    result = _run(upscale_image(str(sample_image), scale=3, mode="ai", arch=BUNDLED_ARCH, output_format="png"))
    out_bgr = cv2.imread(str(result["output_path"]))
    assert out_bgr.shape == (64, 64, 3)


def test_upscale_image_ai_scale_4(sample_image: Path):
    result = _run(upscale_image(str(sample_image), scale=4, mode="ai", arch=BUNDLED_ARCH, output_format="png"))
    out_bgr = cv2.imread(str(result["output_path"]))
    assert out_bgr.shape == (64, 64, 3)


def test_upscale_image_classical_preserves_dimensions(sample_image: Path):
    result = _run(upscale_image(str(sample_image), mode="classical", output_format="png"))
    out_bgr = cv2.imread(str(result["output_path"]))
    assert out_bgr.shape == (32, 32, 3)


def test_upscale_image_multi_format(sample_image: Path):
    for fmt in ["png", "jpeg", "webp", "tiff"]:
        result = _run(
            upscale_image(
                str(sample_image),
                mode="classical",
                output_format=fmt,
            )
        )
        out_path = Path(result["output_path"])
        assert out_path.is_file()
        expected = ".jpg" if fmt == "jpeg" else f".{fmt}"
        assert out_path.suffix.lower() == expected


def test_upscale_image_mcp_call_tool_end_to_end(sample_image: Path):
    """Exercise the exact in-process path FastMCP uses for Claude Desktop / Cursor."""
    content_blocks, structured = _run(
        mcp.call_tool(
            "upscale_image",
            {
                "image_path": str(sample_image),
                "scale": 2,
                "arch": BUNDLED_ARCH,
                "mode": "ai",
                "output_format": "png",
            },
        )
    )
    assert isinstance(content_blocks, list) and len(content_blocks) >= 1
    assert Path(structured["output_path"]).is_file()


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


def test_upscale_image_invalid_arch_raises(sample_image: Path):
    with pytest.raises(ValueError, match="Unsupported arch"):
        _run(upscale_image(str(sample_image), arch="nonexistent"))


def test_upscale_image_scale_zero_raises(sample_image: Path):
    with pytest.raises(ValueError):
        _run(upscale_image(str(sample_image), scale=0))


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def test_health_tool():
    _, structured = _run(mcp.call_tool("health", {}))
    assert structured["version"] == "0.4.0"
    assert "providers_available" in structured
    assert "bundled_models" in structured


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_tool():
    _, structured = _run(mcp.call_tool("list_models", {}))
    assert structured["default_arch"] == "edsr_baseline"
    assert "models" in structured
    models = structured["models"]
    for arch in ("span", "sr_compact", "realplskr", "edsr_baseline"):
        assert arch in models
        assert "filename" in models[arch]
        assert "approx_size_mb" in models[arch]
        assert "best_for" in models[arch]


# ---------------------------------------------------------------------------
# batch_upscale
# ---------------------------------------------------------------------------

def test_batch_upscale(tmp_path: Path):
    images = []
    for i in range(2):
        img_path = tmp_path / f"sample_{i}.jpg"
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.circle(arr, (8, 8), 4, (0, 255, 0), -1)
        cv2.imwrite(str(img_path), arr)
        images.append(str(img_path))

    results = batch_upscale_tool(images, scale=2, arch=BUNDLED_ARCH, mode="ai")
    assert len(results) == 2
    for entry in results:
        assert entry["status"] == "ok"
        assert "output_path" in entry


def test_batch_upscale_missing_image(tmp_path: Path):
    results = batch_upscale_tool([str(tmp_path / "missing.jpg")], scale=2, arch=BUNDLED_ARCH, mode="ai")
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "error" in results[0]


def batch_upscale_tool(image_paths, scale=2, arch="edsr_baseline", mode="ai", output_format="png", quality=95):
    """Thin wrapper calling the MCP batch_upscale tool via call_tool."""
    _, structured = _run(
        mcp.call_tool(
            "batch_upscale",
            {
                "image_paths": image_paths,
                "scale": scale,
                "arch": arch,
                "mode": mode,
                "output_format": output_format,
                "quality": quality,
            },
        )
    )
    return structured["result"]
