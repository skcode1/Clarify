"""Model Context Protocol (MCP) server interface for clarify-ai.

Exposes the local image upscaler as MCP tools so assistants (Claude Desktop,
Cursor, custom agent runtimes) can enhance images without ever sending image
data to an external cloud. All inference runs locally through ONNX Runtime +
OpenCV.

Run the server over stdio:

    python -m clarify.mcp_server

or via the installed entry point:

    clarify-mcp

Claude Desktop / Cursor configuration (``claude_desktop_config.json``):

    {
      "mcpServers": {
        "clarify": {
          "command": "python",
          "args": ["-m", "clarify.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal, cast

from mcp.server.fastmcp import FastMCP

from clarify.core import (
    DEFAULT_ARCH,
    DEFAULT_SCALE,
    MODEL_PRESETS,
    ModelArch,
    enhance,
    health_check,
)
from clarify.exceptions import ClarifyError

logger = logging.getLogger("clarify.mcp_server")

# Initialize MCP Server
mcp = FastMCP("clarify-ai")

Scale = Literal[2, 3, 4]
OutputFormat = Literal["png", "jpeg", "jpg", "webp", "tiff", "tif"]
EnhanceMode = Literal["ai", "classical", "denoise", "deblur", "face_restore"]
ModelArchLit = Literal["span", "sr_compact", "realplskr", "edsr_baseline"]

AI_MODE_SCALES = (2, 3, 4)
SUPPORTED_OUTPUT_FORMATS = ("png", "jpeg", "jpg", "webp", "tiff", "tif")
SUPPORTED_MODES = ("ai", "classical", "denoise", "deblur", "face_restore")
SUPPORTED_ARCHS = tuple(MODEL_PRESETS.keys())


def _validate_inputs(
    image_path: str,
    scale: int,
    mode: str,
    output_format: str,
    arch: str,
) -> Path:
    """Validate MCP tool arguments and return a resolved, absolute input path."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image_path must be a non-empty string.")

    resolved = Path(image_path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Input image not found on disk: {resolved}")

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode '{mode}'. Choose one of: {', '.join(SUPPORTED_MODES)}."
        )

    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError(f"scale must be a positive integer, got {scale!r}.")

    if mode == "ai" and scale not in AI_MODE_SCALES:
        raise ValueError(
            f"Invalid scale '{scale}' for AI mode. Use one of: {', '.join(map(str, AI_MODE_SCALES))}."
        )

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported arch '{arch}'. Choose one of: {', '.join(SUPPORTED_ARCHS)}."
        )

    fmt = output_format.lower().lstrip(".")
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Choose one of: {', '.join(SUPPORTED_OUTPUT_FORMATS)}."
        )

    return resolved


@mcp.tool()
async def upscale_image(
    image_path: str,
    scale: int = 2,
    arch: ModelArchLit = "span",
    mode: EnhanceMode = "ai",
    output_format: OutputFormat = "png",
    quality: int = 95,
    strength: float = 1.0,
) -> dict[str, Any]:
    """Enhance and upscale an image locally using ONNX Runtime.

    The image is processed entirely on the local machine — no image data is
    uploaded to an external cloud. CPU-intensive ONNX/OpenCV inference runs in
    an executor thread so the async event loop stays responsive.

    Args:
        image_path: Absolute or relative path to the image on disk.
        scale: Upscale factor (2, 3, or 4 for 'ai' mode).
        arch: Model architecture preset ('span', 'sr_compact', 'realplskr', 'edsr_baseline').
        mode: Processing algorithm ('ai', 'classical', 'denoise', 'deblur', 'face_restore').
        output_format: Target format ('png', 'jpeg', 'webp', 'tiff').
        quality: Compression quality (1-100) for JPEG and WebP.
        strength: Filtering strength for denoise/deblur (default: 1.0).

    Returns:
        A dict with the output filepath, arch, scale, and mode used.
    """
    input_path = _validate_inputs(image_path, scale, mode, output_format, arch)

    logger.info(
        "MCP upscale_image start: input=%s mode=%s scale=%s arch=%s format=%s",
        input_path,
        mode,
        scale,
        arch,
        output_format,
    )

    loop = asyncio.get_running_loop()
    out_path = await loop.run_in_executor(
        None,
        lambda: enhance(
            image=str(input_path),
            mode=cast(EnhanceMode, mode),
            scale=cast(Scale, scale),
            arch=cast(ModelArch, arch),
            output_format=output_format,
            quality=quality,
            strength=strength,
        ),
    )

    result_path = Path(out_path).resolve()
    logger.info("MCP upscale_image complete: input=%s output=%s", input_path, result_path)

    return {
        "output_path": str(result_path),
        "input_path": str(input_path),
        "mode": mode,
        "arch": arch if mode == "ai" else None,
        "scale": scale if mode == "ai" else None,
        "format": output_format,
    }


@mcp.tool()
async def health() -> dict[str, Any]:
    """Return clarify-ai runtime health metadata."""
    return health_check()


@mcp.tool()
async def list_models() -> dict[str, Any]:
    """List available model architecture presets with metadata."""
    return {
        "default_arch": DEFAULT_ARCH,
        "default_scale": DEFAULT_SCALE,
        "models": {
            arch: {
                "filename": preset.filename,
                "description": preset.description,
                "approx_size_mb": preset.approx_size_mb,
                "best_for": preset.best_for,
                "default_scale": preset.default_scale,
                "bundled": (Path(__file__).resolve().parent.parent / "models" / preset.filename).is_file(),
            }
            for arch, preset in MODEL_PRESETS.items()
        },
    }


@mcp.tool()
async def batch_upscale(
    image_paths: list[str],
    scale: int = 2,
    arch: ModelArchLit = "span",
    mode: EnhanceMode = "ai",
    output_format: OutputFormat = "png",
    quality: int = 95,
) -> list[dict[str, Any]]:
    """Enhance multiple images in sequence.

    Returns one result dict per input image. Failed items include an ``error``
    field instead of ``output_path``.
    """
    results: list[dict[str, Any]] = []
    for image_path in image_paths:
        entry: dict[str, Any] = {"input_path": image_path}
        try:
            resolved = _validate_inputs(image_path, scale, mode, output_format, arch)
            loop = asyncio.get_running_loop()
            out_path = await loop.run_in_executor(
                None,
                lambda: enhance(
                    image=str(resolved),
                    mode=cast(EnhanceMode, mode),
                    scale=cast(Scale, scale),
                    arch=cast(ModelArch, arch),
                    output_format=output_format,
                    quality=quality,
                ),
            )
            entry["output_path"] = str(Path(out_path).resolve())
            entry["status"] = "ok"
        except (ClarifyError, ValueError, FileNotFoundError) as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            entry["error_code"] = getattr(exc, "code", exc.__class__.__name__)
        results.append(entry)
    return results


def main() -> None:
    """Entry point for the ``clarify-mcp`` console script and ``python -m``."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
