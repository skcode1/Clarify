"""Core enhancement pipeline with device detection and singleton model caching."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional, Union
from PIL import Image, UnidentifiedImageError
from clarify.exceptions import ImageNotFoundError, UnsupportedImageError
_CACHED_MODEL = None
_CACHED_DEVICE = None
_CACHED_SCALE = None
def _resolve_input_path(path_str: Union[str, Path]) -> Path:
    """Resolve image path against calling file or working directory."""
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw
    stack = inspect.stack()
    for frame_info in stack[2:]:
        caller_filename = frame_info.filename
        if (
                caller_filename
                and not caller_filename.startswith("<")
                and Path(caller_filename).is_file()
            ):
                return (Path(caller_filename).parent / raw).resolve()
        return (Path.cwd() / raw).resolve()
def _get_device() -> str:
    """Auto-detect CUDA, Apple Silicon MPS, or fallback to CPU."""
    global _CACHED_DEVICE
    if _CACHED_DEVICE is None:
        try:
            import torch

            if torch.cuda.is_available():
                _CACHED_DEVICE = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _CACHED_DEVICE = "mps"
            else:
                _CACHED_DEVICE = "cpu"
        except ImportError:
            _CACHED_DEVICE = "cpu"
    return _CACHED_DEVICE
def _load_model(scale: int = 2):
    """Fetch model from Hugging Face hub once per session."""
    global _CACHED_MODEL, _CACHED_SCALE
    if _CACHED_MODEL is None or _CACHED_SCALE != scale:
        from super_image import EdsrModel

        device = _get_device()
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)
        model = model.to(device)
        model.eval()
        _CACHED_MODEL = model
        _CACHED_SCALE = scale
    return _CACHED_MODEL
def enhance(
    image: Union[str, Path],
    *,
    scale: int = 2,
    output: Optional[Union[str, Path]] = None,
) -> Path:
    """Enhance and upscale an image using EDSR AI super-resolution.

    Args:
        image: Relative or absolute path to the input image file.
        scale: Upscaling multiplier factor (2, 3, or 4). Default is 2.
        output: Optional custom output path. If None, saves as <stem>_enhanced.png
                next to the source file.

    Returns:
        pathlib.Path: Absolute path to the generated output image.
    """
    import torch
    from super_image import ImageLoader

    input_path = _resolve_input_path(image)
    if not input_path.is_file():
        raise ImageNotFoundError(f"Source image not found: {input_path}")

    if output:
        out_candidate = Path(output).expanduser()
        output_path = (
            out_candidate
            if out_candidate.is_absolute()
            else (input_path.parent / out_candidate).resolve()
        )
    else:
        output_path = input_path.parent / f"{input_path.stem}_enhanced.png"

    try:
        pil_image = Image.open(input_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"Cannot identify or decode image: {input_path}") from exc

    device = _get_device()
    model = _load_model(scale=scale)
    inputs = ImageLoader.load_image(pil_image).to(device)
    with torch.no_grad():
        predictions = model(inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ImageLoader.save_image(predictions.cpu(), str(output_path))

    return output_path