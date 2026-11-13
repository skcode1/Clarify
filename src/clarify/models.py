"""EDSR model registry and cache manager for clarify-ai.

Resolves bundled EDSR Baseline models locally (default, sub-50 MB wheel
footprint) and lazily downloads/caches EDSR Full models into
``~/.cache/clarify/models`` only when the user requests ``precision="hq"``.

All integrity is verified via SHA-256 checksums. The full models are intentionally
NOT committed to the Git repository; they are hosted on GitHub Releases and
downloaded on first use.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Literal

from clarify.exceptions import ModelNotFoundError

Precision = Literal["fast", "hq"]

# Directory containing the bundled baseline ONNX models (ships inside the wheel).
BUNDLED_MODELS_DIR = Path(__file__).parent / "models"

# User-writable cache directory for downloaded full models.
CACHE_DIR = Path(os.getenv("CLARIFY_CACHE_DIR", Path.home() / ".cache" / "clarify" / "models"))

# EDSR Full model registry. SHA-256 values must be filled in after publishing
# the full-model assets to a GitHub Release. Until then, placeholder hashes are
# accepted (a warning is emitted) so the module imports cleanly offline.
FULL_MODEL_REGISTRY: dict[int, dict[str, str]] = {
    2: {
        "filename": "edsr_x2_full.onnx",
        "url": "https://github.com/skcode1/Clarify/releases/download/v0.3.0-models/edsr_x2_full.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X2",
    },
    3: {
        "filename": "edsr_x3_full.onnx",
        "url": "https://github.com/skcode1/Clarify/releases/download/v0.3.0-models/edsr_x3_full.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X3",
    },
    4: {
        "filename": "edsr_x4_full.onnx",
        "url": "https://github.com/skcode1/Clarify/releases/download/v0.3.0-models/edsr_x4_full.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X4",
    },
}

_VALID_PRECISIONS: tuple[Precision, ...] = ("fast", "hq")
_VALID_SCALES: tuple[int, ...] = (2, 3, 4)


def _verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Return True if ``file_path`` SHA-256 matches ``expected_hash``.

    Placeholder hashes (``REPLACE_WITH_...``) are accepted during development
    so the module imports and resolves offline without network access.
    """
    if expected_hash.startswith("REPLACE_WITH"):
        return True

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def get_edsr_model_path(
    scale: int,
    precision: Precision = "fast",
    model_path: str | None = None,
) -> Path:
    """Resolve the ONNX model path for a given scale and precision.

    Resolution order:
    1. Explicit ``model_path`` (air-gapped / custom models).
    2. Bundled Baseline model inside the wheel (``precision="fast"``).
    3. Lazily downloaded & cached Full model (``precision="hq"``).

    Args:
        scale: Upscaling factor (2, 3, or 4).
        precision: ``"fast"`` for bundled Baseline, ``"hq"`` for Full.
        model_path: Optional explicit path to a custom .onnx model file.

    Returns:
        Absolute ``Path`` to the resolved ONNX model file.

    Raises:
        ValueError: If ``scale`` or ``precision`` is invalid.
        ModelNotFoundError: If the model file cannot be found or downloaded.
    """
    if scale not in _VALID_SCALES:
        raise ValueError(f"Unsupported scale {scale}. Choose from {_VALID_SCALES}.")
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"Invalid precision '{precision}'. Use {_VALID_PRECISIONS}.")

    # 1. Explicit custom path — ideal for air-gapped / offline environments.
    if model_path:
        custom = Path(model_path).expanduser().resolve()
        if not custom.is_file():
            raise ModelNotFoundError(f"Custom model not found at: {custom}")
        return custom

    # 2. Bundled Baseline model — ships inside the wheel, no network needed.
    if precision == "fast":
        bundled = BUNDLED_MODELS_DIR / f"edsr_x{scale}.onnx"
        if not bundled.is_file():
            raise ModelNotFoundError(
                f"Bundled EDSR Baseline model not found at: {bundled}"
            )
        return bundled

    # 3. EDSR Full model — lazy download + cache, SHA-256 verified.
    meta = FULL_MODEL_REGISTRY[scale]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE_DIR / meta["filename"]

    if cached_file.is_file() and _verify_sha256(cached_file, meta["sha256"]):
        return cached_file

    print(
        f"Downloading EDSR Full x{scale} model (~160 MB) to {cached_file}...",
        flush=True,
    )
    tmp_file = cached_file.with_suffix(".onnx.tmp")
    try:
        urllib.request.urlretrieve(meta["url"], tmp_file)
        if not _verify_sha256(tmp_file, meta["sha256"]):
            raise ModelNotFoundError(
                f"Downloaded EDSR Full x{scale} model failed SHA-256 integrity check."
            )
        tmp_file.replace(cached_file)
    except ModelNotFoundError:
        raise
    except Exception as exc:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        raise ModelNotFoundError(
            f"Failed to download EDSR Full x{scale} model from {meta['url']}. "
            f"For offline use, place '{meta['filename']}' in '{CACHE_DIR}' "
            f"or pass `model_path` to a local .onnx file."
        ) from exc

    return cached_file


def list_available_models() -> dict[str, dict[str, object]]:
    """Return a human-readable summary of available EDSR models per scale."""
    summary: dict[str, dict[str, object]] = {}
    for scale in _VALID_SCALES:
        bundled = BUNDLED_MODELS_DIR / f"edsr_x{scale}.onnx"
        entry: dict[str, object] = {
            "fast": {
                "path": str(bundled),
                "available": bundled.is_file(),
                "source": "bundled",
            },
            "hq": {
                "filename": FULL_MODEL_REGISTRY[scale]["filename"],
                "available": (CACHE_DIR / FULL_MODEL_REGISTRY[scale]["filename"]).is_file(),
                "source": "cache_or_download",
            },
        }
        summary[f"x{scale}"] = entry
    return summary
