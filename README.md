<div align="center">

# clarify-ai

**Zero-config AI image enhancement and super-resolution for Python — run locally, no PyPI required.**

One function call. No PyPI. No internet after clone for default models. Sub-50 MB footprint.

</div>

---

## What it does

`clarify-ai` enhances and upscales images using ONNX Runtime and OpenCV — no PyTorch, no TensorFlow, no heavy ML frameworks at runtime.

It takes any image and produces a clearer, sharper, higher-quality version:

| Mode | What it does | Model Delivery |
|------|--------------|----------------|
| **ai** (default) | AI super-resolution (EDSR Baseline, ~64ch, 16 blocks) — upscales 2x, 3x, or 4x with clarity polish | Bundled at install — offline by default |
| **ai (hq)** | AI super-resolution (EDSR Full, ~256ch, 32 blocks) — higher quality, downloaded on demand | Lazy per-scale download + cache |
| **classical** | OpenCV detail enhancement + denoising (no upscaling) | No model needed |
| **denoise** | Non-local means + bilateral filtering to remove noise | No model needed |
| **deblur** | LAB-space unsharp masking to recover edges | No model needed |
| **face_restore** | GFPGAN ONNX pipeline to restore faces | Lazy download + cache |

Output formats: **PNG, JPEG, WebP, TIFF** (configurable quality).

---

## Precision Modes (AI Mode Only)

`clarify-ai` ships with **EDSR Baseline** models bundled in the wheel (sub-50 MB footprint, works fully offline with no internet needed). For users who need maximum quality, **EDSR Full** models (~160 MB each) are available via ``precision="hq"`` — they are downloaded lazily and individually on first use, then cached in ``~/.cache/clarify/models`` with SHA-256 integrity verification.

| Precision | Model | Size | Delivery |
|-----------|-------|------|----------|
| ``fast`` (default) | EDSR Baseline | ~5–6 MB each | Bundled at install — offline by default |
| ``hq`` | EDSR Full | ~160 MB each | Downloaded per-scale on first use — cached locally |

### Model Lifecycle for External Users

**Install time** — Only EDSR Baseline models (~5–6 MB each) are bundled. The initial download stays under 50 MB, so `clarify-ai` works offline immediately after install.

**On first use of hq mode** — The EDSR Full model for the requested scale is downloaded individually (e.g., scale=2 downloads only `edsr_x2_full.onnx`, ~160 MB). The SHA-256 hash is verified before caching. Subsequent runs reuse the cached file locally without touching the internet. Other scales (x3, x4) are downloaded only if you explicitly request them later.

**Download safety** — Downloads write to a `.onnx.tmp` file first, verify SHA-256 integrity, then atomically replace the final cache file. Interrupted or corrupt downloads never leave a broken model in place.

```python
# Fast: uses bundled EDSR Baseline (default, offline)
enhance("photo.jpg", mode="ai", scale=2, precision="fast")

# High quality: downloads EDSR Full x2 on first use, caches locally
enhance("photo.jpg", mode="ai", scale=2, precision="hq")

# Air-gapped / offline: point to a pre-transferred local .onnx file
enhance("photo.jpg", mode="ai", scale=2, precision="hq", model_path="/models/edsr_x2_full.onnx")
```
---

## Face Restore (GFPGAN)

`face_restore` uses the GFPGAN ONNX model (~350 MB) for facial enhancement. Like EDSR Full, it is NOT bundled in the wheel (to keep the package under 50 MB). It is downloaded lazily on first use from a GitHub Release, or you can supply a local model file.

```python
# Downloads GFPGAN v1_4 on first use, caches locally
enhance("portrait.png", mode="face_restore", output_format="tiff")

# Air-gapped / offline: point to a pre-transferred local .onnx file
enhance("portrait.png", mode="face_restore", model_path="/models/gfpgan_v1_4.onnx")
```
---

## Memory Safety on Large Images

EDSR Full (~43M parameters, 256 channels) can consume significant RAM on high-resolution images. `clarify-ai` automatically uses **tiled inference** when `precision="hq"` is active:

- Images up to 256×256 are processed in a single pass
- Larger images are split into overlapping 256×256 tiles with weight-normalized blending
- This keeps peak RAM usage bounded (typically <2 GB even for 4K inputs)

Baseline mode uses direct full-image inference and is lightweight enough for most images without tiling.

---

## Install & Use

Because `clarify-ai` is not published on PyPI, install it directly from GitHub:

### 1. Direct Git install (most common)

```bash
pip install git+https://github.com/skcode1/Clarify.git
```

### 2. Clone & local install

```bash
git clone https://github.com/skcode1/Clarify.git
cd Clarify
pip install -e ".[dev]"
```

### 3. Build & transfer offline wheel (air-gapped)

```bash
# On a machine with internet
git clone https://github.com/skcode1/Clarify.git
cd Clarify
pip install build
python -m build

# Transfer dist/clarify_ai-0.4.0-py3-none-any.whl to the offline machine
pip install clarify_ai-0.4.0-py3-none-any.whl
```

### 4. Vendor into your project (no install)

```bash
# Copy the package into your project
git clone https://github.com/skcode1/Clarify.git Clarify-temp
cp -r Clarify-temp/src/clarify ./vendor/
rm -rf Clarify-temp

# Use
import sys
sys.path.insert(0, "vendor")
from clarify import enhance
enhance("photo.jpg")
```

### 5. Clone & run directly

```bash
git clone https://github.com/skcode1/Clarify.git
cd Clarify
python -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python-headless onnxruntime pillow

PYTHONPATH=src python -c "from clarify import enhance; enhance('photo.jpg')"
```

### CLI usage

```bash
# Basic AI upscale (2x PNG, bundled baseline — offline)
clarify photo.jpg

# 4x AI upscale to WebP
clarify photo.jpg --mode ai --scale 4 --format webp --quality 90

# Denoise to JPEG
clarify noisy.jpg --mode denoise --format jpeg --quality 92

# Deblur to TIFF
clarify blurry.jpg --mode deblur --format tiff

# Face restore (downloads GFPGAN on first use)
clarify portrait.png --mode face_restore --format png

# Face restore (air-gapped, local model)
clarify portrait.png --mode face_restore --model-path /models/gfpgan_v1_4.onnx

# Custom output path
clarify photo.jpg --output exports/enhanced.png

# High quality: EDSR Full (downloaded on first use, cached locally)
clarify photo.jpg --mode ai --scale 4 --precision hq

# Air-gapped: custom local .onnx model
clarify photo.jpg --mode ai --scale 2 --precision hq --model-path /models/edsr_x2_full.onnx

# Help
clarify --help
```

### Python API

```python
from clarify import enhance

# AI super-resolution 2x, saved as PNG next to input (bundled baseline — offline)
enhance("photo.jpg", mode="ai", scale=2)

# Denoise to JPEG at quality 92
enhance("noisy.jpg", mode="denoise", output_format="jpeg", quality=92)

# Deblur to WebP
enhance("blurry.jpg", mode="deblur", output_format="webp", quality=90)

# Face restore to TIFF (downloads GFPGAN on first use)
enhance("portrait.png", mode="face_restore", output_format="tiff")

# Face restore (air-gapped)
enhance("portrait.png", mode="face_restore", model_path="/models/gfpgan_v1_4.onnx")

# High quality: EDSR Full (downloaded on first use, cached locally)
enhance("photo.jpg", mode="ai", scale=4, precision="hq")

# Air-gapped: custom local .onnx model
enhance("photo.jpg", mode="ai", scale=2, precision="hq", model_path="/models/edsr_x2_full.onnx")
```

---

## For whom is this useful?

| User | How it helps |
|------|--------------|
| **Designers** | Turn low-res or blurry assets into print-ready, polished images with one command |
| **Developers** | Embed image enhancement in apps without bundling PyTorch/TensorFlow (sub-50 MB) |
| **Photographers** | Denoise, deblur, and upscale photos locally — no cloud upload, no API keys |
| **Teams in air-gapped environments** | Build a wheel once, distribute offline — no internet or PyPI needed at runtime (default models only) |
| **CI/CD pipelines** | Batch-enhance images in automated workflows with the CLI |

---

## Requirements

- Python 3.9+
- Runtime: `numpy`, `opencv-python-headless`, `onnxruntime`, `Pillow`
- Dev: `pytest`, `pytest-timeout`, `mypy`, `ruff`, `build`, `twine`

---

## License

**MIT License** — free for personal and commercial use.
