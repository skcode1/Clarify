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
from typing import Literal, cast

from mcp.server.fastmcp import FastMCP

from clarify.core import enhance

logger = logging.getLogger("clarify.mcp_server")

# Initialize MCP Server
mcp = FastMCP("clarify-ai")

Scale = Literal[2, 3, 4]
OutputFormat = Literal["png", "jpeg", "jpg", "webp", "tiff", "tif"]
EnhanceMode = Literal["ai", "classical", "denoise", "deblur", "face_restore"]

AI_MODE_SCALES = (2, 3, 4)
SUPPORTED_OUTPUT_FORMATS = ("png", "jpeg", "jpg", "webp", "tiff", "tif")
SUPPORTED_MODES = ("ai", "classical", "denoise", "deblur", "face_restore")


def _validate_inputs(
    image_path: str,
    scale: int,
    mode: str,
    output_format: str,
) -> Path:
    """Validate MCP tool arguments and return a resolved, absolute input path.

    Raises:
        ValueError: When any argument is out of the supported domain.
        FileNotFoundError: When the input image does not exist on disk.
    """
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
    mode: EnhanceMode = "ai",
    output_format: OutputFormat = "png",
) -> str:
    """Enhance and upscale an image locally using ONNX Runtime.

    The image is processed entirely on the local machine — no image data is
    uploaded to an external cloud. CPU-intensive ONNX/OpenCV inference runs in
    an executor thread so the async event loop stays responsive.

    Args:
        image_path: Absolute or relative path to the image on disk.
        scale: Upscale factor (2, 3, or 4 for 'ai' mode).
        mode: Processing algorithm ('ai', 'classical', 'denoise', 'deblur', 'face_restore').
        output_format: Target format ('png', 'jpeg', 'webp', 'tiff').

    Returns:
        Absolute filepath to the saved, enhanced image.

    Raises:
        ValueError: When mode, scale, or output_format is unsupported.
        FileNotFoundError: When image_path does not point to an existing file.
        ClarifyError: For any clarify-internal processing failure.
    """
    input_path = _validate_inputs(image_path, scale, mode, output_format)

    logger.info(
        "MCP upscale_image start: input=%s mode=%s scale=%s format=%s",
        input_path,
        mode,
        scale,
        output_format,
    )

    # Run sync OpenCV/ONNX inference in executor thread to keep async event loop responsive
    loop = asyncio.get_running_loop()
    out_path = await loop.run_in_executor(
        None,
        lambda: enhance(
            image=str(input_path),
            mode=mode,
            scale=cast(Scale, scale),
            output_format=output_format,
        ),
    )

    result_path = Path(out_path).resolve()
    logger.info("MCP upscale_image complete: input=%s output=%s", input_path, result_path)

    return str(result_path)


def main() -> None:
    """Entry point for the ``clarify-mcp`` console script and ``python -m``."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
