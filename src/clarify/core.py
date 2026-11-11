"""Instant-execution image enhancement using OpenCV and ONNX Runtime."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from .exceptions import ImageNotFoundError, ModelNotFoundError, UnsupportedImageError

# Global in-memory cache for the ONNX Runtime InferenceSession (per scale)
_CACHED_SESSIONS: dict[int, ort.InferenceSession] = {}
_CACHED_INPUT_NAMES: dict[int, str] = {}


def _resolve_path(path_str: Union[str, Path]) -> Path:
    """Resolve file paths relative to caller frame or current directory."""
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


def _get_model_path(scale: int = 2) -> Path:
    """Locate bundled ONNX model file within package directory."""
    model_file = Path(__file__).parent / "models" / f"edsr_x{scale}.onnx"
    if not model_file.is_file():
        raise ModelNotFoundError(f"Model file not found at: {model_file}")
    return model_file


def _get_onnx_session(scale: int = 2) -> tuple[ort.InferenceSession, str]:
    """Retrieve or initialize the ONNX Runtime session for a given scale."""
    global _CACHED_SESSIONS, _CACHED_INPUT_NAMES

    if scale not in _CACHED_SESSIONS:
        model_path = _get_model_path(scale=scale)

        available_providers = ort.get_available_providers()
        # Priority: CUDA (NVIDIA) -> CoreML (Apple Silicon MPS) -> CPU
        providers = [
            p for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            if p in available_providers
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _CACHED_SESSIONS[scale] = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        _CACHED_INPUT_NAMES[scale] = _CACHED_SESSIONS[scale].get_inputs()[0].name

    return _CACHED_SESSIONS[scale], _CACHED_INPUT_NAMES[scale]


def _enhance_classical(img_bgr: np.ndarray) -> np.ndarray:
    """Ultra-fast non-AI detail sharpening and adaptive equalization."""
   
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
    
    
    enhanced = cv2.detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
    return enhanced


def _enhance_ai(img_bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    """Super-resolution reconstruction via ONNX Runtime."""
    session, input_name = _get_onnx_session(scale=scale)

   
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

   
    blob = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...])

    
    ort_outputs = session.run(None, {input_name: blob})

    
    out_tensor = ort_outputs[0][0]  # Shape: (C, H, W)
    out_tensor = np.clip(out_tensor * 255.0, 0, 255).astype(np.uint8)
    out_rgb = np.transpose(out_tensor, (1, 2, 0))  # Shape: (H, W, C)

    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)  # type: ignore[no-any-return]


def enhance(
    image: Union[str, Path],
    *,
    mode: Literal["ai", "classical"] = "ai",
    scale: Literal[2, 3, 4] = 2,
    output: Optional[Union[str, Path]] = None,
) -> Path:
    """Enhance and upscale an image instantly using ONNX Runtime.

    Args:
        image: Path to the target image file.
        mode: "ai" (EDSR super-resolution) or "classical" (OpenCV detail enhancement).
        scale: Upscale multiplier factor (2, 3, or 4). Only applies to "ai" mode.
        output: Optional destination path. Defaults to <stem>_enhanced.png next to source.

    Returns:
        pathlib.Path: Absolute path pointing to the enhanced image file.
    """
    if scale not in (2, 3, 4):
        raise ValueError(f"Invalid scale '{scale}'. Choose 2, 3, or 4.")
        
    input_path = _resolve_path(image)
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

   
    data = np.fromfile(str(input_path), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise UnsupportedImageError(f"Could not decode image at path: {input_path}")

    
    if mode == "ai":
        result_bgr = _enhance_ai(img_bgr, scale=scale)
    elif mode == "classical":
        result_bgr = _enhance_classical(img_bgr)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'ai' or 'classical'.")

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", result_bgr)
    if not success:
        raise UnsupportedImageError(f"Failed to encode enhanced image to {output_path}")

    encoded.tofile(str(output_path))
    return output_path