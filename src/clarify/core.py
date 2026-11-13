"""clarify-ai: Unified high-throughput image enhancement orchestrator."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from clarify.exceptions import (
    ConfigurationError,
    ImageNotFoundError,
    ModelNotFoundError,
    UnsupportedImageError,
)
from clarify.models import Precision, get_edsr_model_path

OutputFormat = Literal["png", "jpeg", "jpg", "webp", "tiff", "tif"]
EnhanceMode = Literal["ai", "classical", "denoise", "deblur", "face_restore"]

# Global in-memory cache for ONNX Runtime InferenceSession (per scale + precision)
_CACHED_SESSIONS: dict[tuple[int, Precision], ort.InferenceSession] = {}
_CACHED_INPUT_NAMES: dict[tuple[int, Precision], str] = {}


def _resolve_path(path_str: str | Path) -> Path:
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


def _get_edsr_session(
    scale: int = 2,
    precision: Precision = "fast",
    model_path: str | None = None,
) -> ort.InferenceSession:
    """Retrieve or initialize the ONNX Runtime session for a given scale + precision.

    Sessions are cached per (scale, precision) tuple so fast (baseline) and hq
    (full) models can coexist without collision.
    """
    global _CACHED_SESSIONS, _CACHED_INPUT_NAMES  # noqa: PLW0602

    cache_key: tuple[int, Precision] = (scale, precision)

    if cache_key not in _CACHED_SESSIONS:
        resolved_model = get_edsr_model_path(
            scale=scale,
            precision=precision,
            model_path=model_path,
        )

        available_providers = ort.get_available_providers()
        providers = [
            p
            for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            if p in available_providers
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        _CACHED_SESSIONS[cache_key] = ort.InferenceSession(
            str(resolved_model), sess_options, providers=providers
        )
        _CACHED_INPUT_NAMES[cache_key] = _CACHED_SESSIONS[cache_key].get_inputs()[0].name

    return _CACHED_SESSIONS[cache_key]


def _encode_and_save(
    img_bgr: np.ndarray,
    target_path: Path,
    out_format: OutputFormat,
    quality: int = 95,
) -> Path:
    """Encodes and saves the matrix with format-specific compression parameters."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_format.lower().lstrip(".")
    params = []

    if fmt in ("jpg", "jpeg"):
        clamped_q = max(1, min(100, int(quality)))
        params = [cv2.IMWRITE_JPEG_QUALITY, clamped_q]
        ext = ".jpg"
    elif fmt == "webp":
        clamped_q = max(1, min(100, int(quality)))
        params = [cv2.IMWRITE_WEBP_QUALITY, clamped_q]
        ext = ".webp"
    elif fmt in ("tif", "tiff"):
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        ext = ".tiff"
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"
    else:
        raise ConfigurationError(f"Unsupported output format: {out_format}")

    final_output_path = target_path.with_suffix(ext)

    success, encoded = cv2.imencode(ext, img_bgr, params)
    if not success or encoded is None:
        raise UnsupportedImageError(f"Failed to encode image to {final_output_path}")

    encoded.tofile(str(final_output_path))
    return final_output_path


def _polish(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Polish pipeline: maximum clean contrast + unsharp + saturation boost.

    Pushes clarity to the highest clean level before halo/artifact thresholds.
    Tuned so AI mode at strength=2.0 produces contrast=1.15, unsharp=1.7, sat=1.15
    — the greatest values that remain artifact-free on portrait/beauty imagery.
    """
    result = img_bgr

    # Aggressive but clean contrast lift via LAB gamma curve
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.float32) * (1.0 + 0.075 * strength), 0, 255).astype(np.uint8)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Strong unsharp mask for edge definition (tuned to stay below halo threshold)
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=0.8)
    result = cv2.addWeighted(result, 1.0 + 0.35 * strength, blurred, -0.35 * strength, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Rich saturation boost in HSV
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * (1.0 + 0.075 * strength), 0, 255).astype(np.uint8)
    result = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    return result


def _run_tiled_edsr(
    session: ort.InferenceSession,
    input_name: str,
    img_chw: np.ndarray,
    scale: int,
    tile_size: int = 256,
    overlap: int = 16,
) -> np.ndarray:
    """Run ONNX super-resolution in overlapping tiles to keep RAM usage low.

    EDSR Full (~43M parameters) can spike RAM by several gigabytes on CPU when
    processing a full 1080p image in one pass. Tiling splits the input into
    overlapping tiles, infers each independently, and blends the results with
    weight normalization to avoid seam artifacts.
    """
    _, c, h, w = img_chw.shape

    # Run directly if the image is small enough to fit in one tile.
    if h <= tile_size and w <= tile_size:
        return session.run(None, {input_name: img_chw})[0]  # type: ignore[no-any-return]

    out_h, out_w = h * scale, w * scale
    output = np.zeros((1, c, out_h, out_w), dtype=np.float32)
    weight = np.zeros((1, c, out_h, out_w), dtype=np.float32)

    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y, max(0, h - tile_size))
            x1 = min(x, max(0, w - tile_size))
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            tile = img_chw[:, :, y1:y2, x1:x2]
            pred = session.run(None, {input_name: tile})[0]

            oy1, oy2 = y1 * scale, y2 * scale
            ox1, ox2 = x1 * scale, x2 * scale

            output[:, :, oy1:oy2, ox1:ox2] += pred
            weight[:, :, oy1:oy2, ox1:ox2] += 1.0

    return output / np.maximum(weight, 1e-8)


def _enhance_ai(
    img_bgr: np.ndarray,
    scale: int = 2,
    polish: bool = True,
    precision: Precision = "fast",
    model_path: str | None = None,
) -> np.ndarray:
    """Super-resolution reconstruction via ONNX Runtime with clarity polish.

    Uses direct full-image inference for ``precision="fast"`` (baseline) and
    tiled inference for ``precision="hq"`` (full) to keep RAM usage bounded on
    large images.
    """
    session = _get_edsr_session(scale=scale, precision=precision, model_path=model_path)
    input_name = _CACHED_INPUT_NAMES[(scale, precision)]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...])

    if precision == "hq":
        out_tensor = _run_tiled_edsr(session, input_name, blob, scale=scale)[0]
    else:
        out_tensor = session.run(None, {input_name: blob})[0][0]

    out_tensor = np.clip(out_tensor * 255.0, 0, 255).astype(np.uint8)
    out_rgb = np.transpose(out_tensor, (1, 2, 0))
    result = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    if polish:
        result = _polish(result, strength=2.0)
    return result  # type: ignore[no-any-return]


def _enhance_classical(img_bgr: np.ndarray) -> np.ndarray:
    """Classical enhancement: denoise + detail sharpening + contrast polish."""
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
    enhanced = cv2.detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
    return _polish(enhanced, strength=0.4)


def _enhance_denoise(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Non-Local Means Denoising with edge-preserving bilateral smoothing + polish."""
    h_lum = 7.0 * strength
    h_chroma = 10.0 * strength

    nlm_denoised = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=h_lum,
        hColor=h_chroma,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    bilateral = cv2.bilateralFilter(nlm_denoised, d=5, sigmaColor=35, sigmaSpace=35)
    return _polish(bilateral, strength=0.4)


def _enhance_deblur(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """LAB-space unsharp deconvolution + polish for edge recovery."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    gaussian = cv2.GaussianBlur(l_chan, (0, 0), sigmaX=2.0)

    weight = 1.0 + (0.8 * strength)
    sharpened_l = cv2.addWeighted(l_chan, weight, gaussian, -(weight - 1.0), 0)

    sharpened_l = np.clip(sharpened_l, 0, 255).astype(np.uint8)

    merged = cv2.merge((sharpened_l, a_chan, b_chan))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return _polish(result, strength=strength)


def _restore_face_patch(face_rgb_512: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """Runs a single aligned 512x512 face crop through GFPGAN ONNX."""
    normalized = (face_rgb_512.astype(np.float32) / 127.5) - 1.0
    blob = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[np.newaxis, ...])

    input_name = session.get_inputs()[0].name
    ort_outs = session.run(None, {input_name: blob})

    out_tensor = ort_outs[0][0]
    out_tensor = np.clip((out_tensor + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return np.transpose(out_tensor, (1, 2, 0))


def _enhance_face_restore(img_bgr: np.ndarray) -> np.ndarray:
    """Detects and restores facial areas, then blends back into the image."""
    model_path = Path(__file__).parent / "models" / "gfpgan_v1_4.onnx"
    if not model_path.is_file():
        raise ModelNotFoundError(f"Face restore model missing: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    h, w = img_bgr.shape[:2]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return _polish(cv2.detailEnhance(img_bgr, sigma_s=10, sigma_r=0.15), strength=0.4)

    result = img_bgr.copy()
    for (x, y, fw, fh) in faces:
        margin = int(0.2 * fw)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + fw + margin)
        y2 = min(h, y + fh + margin)

        face_roi = result[y1:y2, x1:x2]
        orig_h, orig_w = face_roi.shape[:2]

        face_roi_512 = cv2.resize(face_roi, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        face_roi_rgb = cv2.cvtColor(face_roi_512, cv2.COLOR_BGR2RGB)

        restored_rgb = _restore_face_patch(face_roi_rgb, session)
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

        restored_orig_size = cv2.resize(restored_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        result[y1:y2, x1:x2] = restored_orig_size

    return _polish(result, strength=0.4)


def enhance(
    image: str | Path,
    *,
    mode: EnhanceMode = "ai",
    scale: Literal[2, 3, 4] = 2,
    precision: Precision = "fast",
    model_path: str | None = None,
    output: str | Path | None = None,
    output_format: OutputFormat = "png",
    quality: int = 95,
    strength: float = 1.0,
) -> Path:
    """Production-grade image enhancement and super-resolution.

    Args:
        image: Local filepath to source image.
        mode: Processing algorithm: 'ai', 'classical', 'denoise', 'deblur', 'face_restore'.
        scale: Upscaling factor (2, 3, 4) for 'ai' mode.
        precision: AI model precision — ``"fast"`` (bundled EDSR Baseline) or
            ``"hq"`` (EDSR Full, lazily downloaded and cached).
        model_path: Optional explicit path to a custom .onnx model file
            (air-gapped / offline use).
        output: Explicit destination path. If None, saves next to source.
        output_format: Target file format ('png', 'jpeg', 'webp', 'tiff').
        quality: Compression quality (1-100) for JPEG and WebP.
        strength: Multiplier factor for filter intensity in classical modes.

    Returns:
        pathlib.Path: Absolute path pointing to the generated file.
    """
    input_path = _resolve_path(image)
    if not input_path.is_file():
        raise ImageNotFoundError(f"Input file not found: {input_path}")

    data = np.fromfile(str(input_path), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise UnsupportedImageError(f"Cannot decode image: {input_path}")

    if mode == "ai":
        if scale not in (2, 3, 4):
            raise ConfigurationError(f"Invalid scale '{scale}' for AI mode. Use 2, 3, or 4.")
        result = _enhance_ai(
            img_bgr,
            scale=scale,
            polish=True,
            precision=precision,
            model_path=model_path,
        )
    elif mode == "classical":
        result = _enhance_classical(img_bgr)
    elif mode == "denoise":
        result = _enhance_denoise(img_bgr, strength=strength)
    elif mode == "deblur":
        result = _enhance_deblur(img_bgr, strength=strength)
    elif mode == "face_restore":
        result = _enhance_face_restore(img_bgr)
    else:
        raise ConfigurationError(f"Invalid mode '{mode}'. Choose 'ai', 'classical', 'denoise', 'deblur', or 'face_restore'.")

    if output:
        dest_candidate = Path(output).expanduser()
        dest_path = dest_candidate if dest_candidate.is_absolute() else (input_path.parent / dest_candidate).resolve()
    else:
        dest_path = input_path.parent / f"{input_path.stem}_{mode}"

    return _encode_and_save(result, dest_path, out_format=output_format, quality=quality)
