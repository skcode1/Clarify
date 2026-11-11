<div align="center">

# clarify-ai

**Zero-config AI image enhancement and super-resolution for Python — run locally, no PyPI required.**

One function call. No PyPI. No internet after clone. Sub-50 MB footprint.

</div>

---

## What it does

`clarify-ai` enhances and upscales images using ONNX Runtime and OpenCV — no PyTorch, no TensorFlow, no heavy ML frameworks at runtime.

It takes any image and produces a clearer, sharper, higher-quality version:

| Mode | What it does |
|------|--------------|
| **ai** | AI super-resolution (EDSR) — upscales 2x, 3x, or 4x with clarity polish |
| **classical** | OpenCV detail enhancement + denoising (no upscaling) |
| **denoise** | Non-local means + bilateral filtering to remove noise |
| **deblur** | LAB-space unsharp masking to recover edges |
| **face_restore** | GFPGAN ONNX pipeline to restore faces |

Output formats: **PNG, JPEG, WebP, TIFF** (configurable quality).

---

## Quick example

```python
from clarify import enhance

# AI super-resolution 2x, saved as PNG next to input
enhance("photo.jpg", mode="ai", scale=2)

# Denoise to JPEG at quality 92
enhance("noisy.jpg", mode="denoise", output_format="jpeg", quality=92)

# Deblur to WebP
enhance("blurry.jpg", mode="deblur", output_format="webp", quality=90)

# Face restore to TIFF
enhance("portrait.png", mode="face_restore", output_format="tiff")
```

---

## Install & Use

### 1. pip (from source)

```bash
git clone https://github.com/your-username/clarify.git
cd clarify
pip install -e ".[dev]"
```

Then:

```python
from clarify import enhance
enhance("photo.jpg")
```

Or via CLI:

```bash
clarify photo.jpg --mode ai --scale 4 --format png
```

### 2. pip (install wheel)

```bash
# Build
cd clarify
pip install build
python -m build

# Install the wheel
pip install dist/clarify_ai-0.3.0-py3-none-any.whl

# Use
clarify photo.jpg
```

### 3. Git (direct dependency)

```bash
pip install git+https://github.com/your-username/clarify.git
```

### 4. Vendor into your project (no install)

```bash
# Copy the package into your project
git clone https://github.com/your-username/clarify.git clarify-temp
cp -r clarify-temp/src/clarify ./vendor/
rm -rf clarify-temp

# Use
import sys
sys.path.insert(0, "vendor")
from clarify import enhance
enhance("photo.jpg")
```

### 5. curl + pip (air-gapped / offline)

```bash
# On a machine with internet
curl -L https://github.com/your-username/clarify/archive/refs/heads/main.tar.gz -o clarify.tar.gz
tar -xzf clarify.tar.gz
cd clarify-main
pip install -e ".[dev]"

# Copy the clarify_ai-x.x.x-py3-none-any.whl to your offline machine
pip install clarify_ai-0.3.0-py3-none-any.whl
```

### 6. GitHub (clone & run directly)

```bash
git clone https://github.com/your-username/clarify.git
cd clarify
python -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python-headless onnxruntime pillow

PYTHONPATH=src python -c "from clarify import enhance; enhance('photo.jpg')"
```

### CLI usage

```bash
# Basic AI upscale (2x PNG)
clarify photo.jpg

# 4x AI upscale to WebP
clarify photo.jpg --mode ai --scale 4 --format webp --quality 90

# Denoise to JPEG
clarify noisy.jpg --mode denoise --format jpeg --quality 92

# Deblur to TIFF
clarify blurry.jpg --mode deblur --format tiff

# Face restore
clarify portrait.png --mode face_restore --format png

# Custom output path
clarify photo.jpg --output exports/enhanced.png

# Help
clarify --help
```

---

## For whom is this useful?

| User | How it helps |
|------|--------------|
| **Designers** | Turn low-res or blurry assets into print-ready, polished images with one command |
| **Developers** | Embed image enhancement in apps without bundling PyTorch/TensorFlow (sub-50 MB) |
| **Photographers** | Denoise, deblur, and upscale photos locally — no cloud upload, no API keys |
| **Teams in air-gapped environments** | Build a wheel once, distribute offline — no internet or PyPI needed at runtime |
| **CI/CD pipelines** | Batch-enhance images in automated workflows with the CLI |

---

## Requirements

- Python 3.9+
- Runtime: `numpy`, `opencv-python-headless`, `onnxruntime`, `Pillow`
- Dev: `pytest`, `pytest-timeout`, `build`, `twine`

---

## License

**MIT License** — free for personal and commercial use.
