"""EDSR model registry and cache manager for clarify-ai.

Resolves bundled EDSR Baseline models locally (default, sub-50 MB wheel
footprint) and lazily downloads/caches EDSR Full and GFPGAN models into
``~/.cache/clarify/models`` only when the user requests them.

All integrity is verified via SHA-256 checksums. The full/GFPGAN models are
intentionally NOT committed to the Git repository; they are hosted on GitHub
Releases and downloaded on first use.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Literal

import onnxruntime as ort  # type: ignore[import-untyped]

from clarify.exceptions import ModelNotFoundError

Precision = Literal["fast", "hq"]

# Directory containing the bundled baseline ONNX models (ships inside the wheel).
BUNDLED_MODELS_DIR = Path(__file__).parent / "models"

# User-writable cache directory for downloaded full/GFPGAN models.
CACHE_DIR = Path(os.getenv("CLARIFY_CACHE_DIR", Path.home() / ".cache" / "clarify" / "models"))

# Package version — kept in sync with pyproject.toml. Used to construct
# GitHub Release download URLs for EDSR Full and GFPGAN models.
__version__ = "0.4.0"

# Expected input/output tensor names for all clarify-ai ONNX models.
EXPECTED_INPUT_NAME = "input"
EXPECTED_OUTPUT_NAME = "output"

# EDSR Full model registry. SHA-256 values must be filled in after publishing
# the full-model assets to a GitHub Release. Replace each placeholder with the
# output of:  sha256sum edsr_x{scale}_full.onnx
FULL_MODEL_REGISTRY: dict[int, dict[str, str]] = {
    2: {
        "filename": "edsr_x2_full.onnx",
        "url": f"https://github.com/skcode1/Clarify/releases/download/v{__version__}-models/edsr_x2_full.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X2",
    },
    3: {
        "filename": "edsr_x3_full.onnx",
        "url": f"https://github.com/skcode1/Clarify/releases/download/v{__version__}-models/edsr_x3_full.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X3",
    },
    4: {
        "filename": "edsr_x4_full.onnx",
        "url": f"https://github.com/skcode1/Clarify/releases/download/v{__version__}-models/edsr_x4_full.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_X4",
    },
}

# GFPGAN face restoration model registry. SHA-256 values must be filled in
# after publishing the GFPGAN asset to a GitHub Release.
GFPGAN_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "v1_4": {
        "filename": "gfpgan_v1_4.onnx",
        "url": f"https://github.com/skcode1/Clarify/releases/download/v{__version__}-models/gfpgan_v1_4.onnx",
        "sha256": "REPLACE_WITH_ACTUAL_SHA256_FOR_GFPGAN_V1_4",
    },
}

_VALID_PRECISIONS: tuple[Precision, ...] = ("fast", "hq")
_VALID_SCALES: tuple[int, ...] = (2, 3, 4)
_VALID_GFPGAN_VERSIONS: tuple[str, ...] = ("v1_4",)

# Download timeout in seconds (160-350 MB models on typical connections).
DOWNLOAD_TIMEOUT_SECONDS = 300


def _is_placeholder_hash(expected_hash: str) -> bool:
    """Return True if the hash is a development placeholder."""
    return expected_hash.startswith("REPLACE_WITH")


def _verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Return True if ``file_path`` SHA-256 matches ``expected_hash``.

    Placeholder hashes (``REPLACE_WITH_...``) are rejected in production
    because they would accept any downloaded content (including 404 HTML pages)
    as a valid model. Users must replace placeholders with real hashes before
    using precision="hq".
    """
    if _is_placeholder_hash(expected_hash):
        return False

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def _validate_onnx_model(model_path: Path) -> None:
    """Validate that an ONNX model has the expected input/output tensor names.

    Raises:
        ModelNotFoundError: If the model cannot be loaded or has unexpected
            tensor names/shapes.
    """
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ModelNotFoundError(
            f"Failed to load ONNX model at {model_path}: {exc}"
        ) from exc

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    if EXPECTED_INPUT_NAME not in input_names:
        raise ModelNotFoundError(
            f"ONNX model at {model_path} has unexpected input tensor names: {input_names}. "
            f"Expected '{EXPECTED_INPUT_NAME}'."
        )
    if EXPECTED_OUTPUT_NAME not in output_names:
        raise ModelNotFoundError(
            f"ONNX model at {model_path} has unexpected output tensor names: {output_names}. "
            f"Expected '{EXPECTED_OUTPUT_NAME}'."
        )


def _download_file(url: str, dest: Path, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    """Download a file from ``url`` to ``dest`` with timeout and progress.

    Uses ``urllib.request.urlopen`` (not the deprecated ``urlretrieve``) so the
    download works reliably on Python 3.12+ and can be interrupted cleanly.

    Raises:
        ModelNotFoundError: If the download fails for any reason.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise ModelNotFoundError(
                    f"Server returned HTTP {response.status} for {url}. "
                    f"Check that the GitHub Release tag and assets exist."
                )
            total_size = response.headers.get("Content-Length")
            if total_size is not None:
                total_size = int(total_size)

            downloaded = 0
            with open(dest, "wb") as out_file:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded * 100 // total_size
                        if percent % 10 == 0:
                            print(f"  Downloaded {downloaded:,} / {total_size:,} bytes ({percent}%)", flush=True)
    except ModelNotFoundError:
        raise
    except Exception as exc:
        raise ModelNotFoundError(
            f"Failed to download from {url}: {exc}"
        ) from exc


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
        _validate_onnx_model(custom)
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

    # Reject placeholder hashes so users don't accidentally accept 404 HTML.
    if _is_placeholder_hash(meta["sha256"]):
        raise ModelNotFoundError(
            f"EDSR Full x{scale} model is not yet configured for download. "
            f"The SHA-256 hash for '{meta['filename']}' is still a placeholder "
            f"('{meta['sha256']}'). "
            f"To use precision='hq', either:\n"
            f"  • Replace the placeholder hash in src/clarify/models.py with the "
            f"    actual SHA-256 of the published model, or\n"
            f"  • Pass a local .onnx file via model_path='/path/to/model.onnx', or\n"
            f"  • Use precision='fast' for the bundled Baseline model."
        )

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
        _download_file(meta["url"], tmp_file)
        if not _verify_sha256(tmp_file, meta["sha256"]):
            tmp_file.unlink(missing_ok=True)
            raise ModelNotFoundError(
                f"Downloaded EDSR Full x{scale} model failed SHA-256 integrity check. "
                f"The file may be corrupted or the published hash may not match."
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


def get_gfpgan_model_path(
    version: str = "v1_4",
    model_path: str | None = None,
) -> Path:
    """Resolve the GFPGAN ONNX model path.

    Resolution order:
    1. Explicit ``model_path`` (air-gapped / custom models).
    2. Bundled GFPGAN model inside the wheel (if present).
    3. Lazily downloaded & cached GFPGAN model from GitHub Release.

    Args:
        version: GFPGAN model version (currently only ``"v1_4"``).
        model_path: Optional explicit path to a custom .onnx model file.

    Returns:
        Absolute ``Path`` to the resolved GFPGAN ONNX model file.

    Raises:
        ValueError: If ``version`` is invalid.
        ModelNotFoundError: If the model file cannot be found or downloaded.
    """
    if version not in _VALID_GFPGAN_VERSIONS:
        raise ValueError(f"Unsupported GFPGAN version '{version}'. Use {_VALID_GFPGAN_VERSIONS}.")

    # 1. Explicit custom path — ideal for air-gapped / offline environments.
    if model_path:
        custom = Path(model_path).expanduser().resolve()
        if not custom.is_file():
            raise ModelNotFoundError(f"Custom GFPGAN model not found at: {custom}")
        _validate_onnx_model(custom)
        return custom

    # 2. Bundled GFPGAN model inside the wheel (if present).
    bundled = BUNDLED_MODELS_DIR / f"gfpgan_{version}.onnx"
    if bundled.is_file():
        return bundled

    # 3. Lazily downloaded & cached GFPGAN model — SHA-256 verified.
    meta = GFPGAN_MODEL_REGISTRY[version]

    # Reject placeholder hashes so users don't accidentally accept 404 HTML.
    if _is_placeholder_hash(meta["sha256"]):
        raise ModelNotFoundError(
            f"GFPGAN {version} model is not yet configured for download. "
            f"The SHA-256 hash for '{meta['filename']}' is still a placeholder "
            f"('{meta['sha256']}'). "
            f"To use mode='face_restore', either:\n"
            f"  • Replace the placeholder hash in src/clarify/models.py with the "
            f"    actual SHA-256 of the published model, or\n"
            f"  • Pass a local .onnx file via model_path='/path/to/gfpgan.onnx'."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE_DIR / meta["filename"]

    if cached_file.is_file() and _verify_sha256(cached_file, meta["sha256"]):
        return cached_file

    print(
        f"Downloading GFPGAN {version} model (~350 MB) to {cached_file}...",
        flush=True,
    )
    tmp_file = cached_file.with_suffix(".onnx.tmp")
    try:
        _download_file(meta["url"], tmp_file)
        if not _verify_sha256(tmp_file, meta["sha256"]):
            tmp_file.unlink(missing_ok=True)
            raise ModelNotFoundError(
                f"Downloaded GFPGAN {version} model failed SHA-256 integrity check. "
                f"The file may be corrupted or the published hash may not match."
            )
        tmp_file.replace(cached_file)
    except ModelNotFoundError:
        raise
    except Exception as exc:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        raise ModelNotFoundError(
            f"Failed to download GFPGAN {version} model from {meta['url']}. "
            f"For offline use, place '{meta['filename']}' in '{CACHE_DIR}' "
            f"or pass `model_path` to a local .onnx file."
        ) from exc

    return cached_file


def list_available_models() -> dict[str, dict[str, object]]:
    """Return a human-readable summary of available EDSR and GFPGAN models."""
    summary: dict[str, dict[str, object]] = {}
    for scale in _VALID_SCALES:
        bundled = BUNDLED_MODELS_DIR / f"edsr_x{scale}.onnx"
        meta = FULL_MODEL_REGISTRY[scale]
        hq_cached = CACHE_DIR / meta["filename"]
        entry: dict[str, object] = {
            "fast": {
                "path": str(bundled),
                "available": bundled.is_file(),
                "source": "bundled",
            },
            "hq": {
                "filename": meta["filename"],
                "available": hq_cached.is_file() and _verify_sha256(hq_cached, meta["sha256"]),
                "source": "cache_or_download",
                "hash_configured": not _is_placeholder_hash(meta["sha256"]),
            },
        }
        summary[f"x{scale}"] = entry

    # GFPGAN entry
    gfpgan_meta = GFPGAN_MODEL_REGISTRY["v1_4"]
    gfpgan_cached = CACHE_DIR / gfpgan_meta["filename"]
    gfpgan_bundled = BUNDLED_MODELS_DIR / "gfpgan_v1_4.onnx"
    summary["face_restore"] = {
        "gfpgan_v1_4": {
            "path": str(gfpgan_bundled),
            "available": gfpgan_bundled.is_file() or (gfpgan_cached.is_file() and _verify_sha256(gfpgan_cached, gfpgan_meta["sha256"])),
            "source": "bundled" if gfpgan_bundled.is_file() else "cache_or_download",
            "hash_configured": not _is_placeholder_hash(gfpgan_meta["sha256"]),
        }
    }

    return summary
