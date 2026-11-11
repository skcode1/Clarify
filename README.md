<div align="center">

# clarify-ai

**Zero-config AI image upscaler and enhancement engine for Python — run locally, no PyPI required.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="Black">
  <a href="https://mypy-lang.org/">
    <img src="https://img.shields.io/badge/mypy-checked-blue.svg?style=flat-square" alt="Mypy">
  </a>
  <img src="https://img.shields.io/badge/install-local%20only-orange?style=flat-square" alt="Local Install Only">
  <img src="https://img.shields.io/badge/version-0.3.0-purple.svg?style=flat-square" alt="Version">
</p>

**Enhance your images with just one function call — clarity polished to "after"-state quality.**

No manual model configuration. No tensor manipulation. No complicated image-processing pipelines. No internet required after initial clone.

[Features](#-key-features) •
[Quick Install](#-quick-install) •
[Usage](#-usage) •
[CLI](#-command-line-interface) •
[API Reference](#-api-reference) •
[Architecture](#-architecture) •
[Contributing](#-contributing)

</div>

---

## 📌 Overview

`clarify-ai` is a Python image enhancement and super-resolution library designed to make AI-powered image upscaling as simple as possible — **entirely offline, no package index needed**.

Traditionally, implementing AI-based image enhancement requires developers to:
- Configure deep learning frameworks (PyTorch, TensorFlow)
- Download and manage pretrained model weights
- Handle image-to-tensor conversions
- Configure CUDA or other hardware accelerators
- Manage device transfers and inference
- Reconstruct enhanced images
- Handle image encoding and output paths

**clarify-ai abstracts this entire workflow into a simple Python interface** — bundled with ONNX models, zero heavy ML frameworks at runtime, sub-50 MB footprint.

```python
from clarify import enhance

enhanced_path = enhance("photo.jpg")
print(f"Enhanced image saved at: {enhanced_path}")
```

By default, the library:
1. Resolves the input image path (relative to your script)
2. Automatically selects the best available hardware (CUDA → CoreML → CPU)
3. Loads the bundled EDSR super-resolution model
4. Performs AI-powered image upscaling (2x, 3x, or 4x)
5. Applies the **clarity polish pipeline** — maximum clean contrast + unsharp + saturation boost
6. Exports the enhanced image in your chosen format (PNG, JPEG, WebP, TIFF)
7. Returns the absolute output path as a `pathlib.Path` object

**Example project structure:**
```text
my-project/
├── main.py          # your script
├── photo.jpg        # input image
└── photo_ai.png     # generated output (polished, "after"-state quality)
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Zero-config API** | Enhance images with a single function call |
| **AI Super-Resolution** | Pretrained EDSR (Enhanced Deep Residual Networks) bundled |
| **Classical Enhancement** | OpenCV-based denoising + detail enhancement (no upscaling) |
| **AI Denoise** | Non-Local Means + bilateral filtering for noise removal |
| **AI Deblur** | LAB-space unsharp masking for edge recovery |
| **Face Restore** | GFPGAN ONNX pipeline for facial restoration |
| **Clarity Polish Pipeline** | Maximum clean contrast + unsharp + saturation boost on every output |
| **Auto Hardware Detection** | CUDA (NVIDIA) → CoreML (Apple Silicon) → CPU fallback |
| **In-Memory Model Caching** | Reuses loaded models across calls |
| **Smart Path Resolution** | Relative paths resolve against calling script's directory |
| **Multiple Upscaling Factors** | 2x, 3x, 4x (AI mode only) |
| **Multi-Format Output** | PNG, JPEG (quality), WebP (quality), TIFF |
| **Custom Output Paths** | Specify any destination |
| **CLI Included** | `clarify` terminal command |
| **Fully Typed** | `py.typed` + mypy strict compliance |
| **No Runtime Downloads** | All models bundled |
| **No Heavy ML Frameworks** | PyTorch-free at runtime (<50 MB wheel) |

---

## 🚀 Quick Install

### Option 1: Clone & Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/clarify.git
cd clarify

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode (includes dev dependencies)
pip install -e ".[dev]"
```

### Option 2: Clone & Use Directly (No Install)

```bash
git clone https://github.com/your-username/clarify.git
cd clarify
python -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python-headless onnxruntime pillow

# Use directly from source
PYTHONPATH=src python -c "from clarify import enhance; print('Ready!')"
```

### Option 3: Vendor into Your Project

Copy the `src/clarify` directory into your project:

```bash
# In your project root
git clone https://github.com/your-username/clarify.git clarify-temp
cp -r clarify-temp/src/clarify ./vendor/
rm -rf clarify-temp
```

Then use it:
```python
import sys
sys.path.insert(0, "vendor")
from clarify import enhance

enhance("photo.jpg")
```

### Option 4: Build & Install Wheel (Air-gapped Environments)

```bash
git clone https://github.com/your-username/clarify.git
cd clarify
pip install build
python -m build  # Creates dist/clarify_ai-0.3.0-py3-none-any.whl

# Copy .whl to target machine and install
pip install clarify_ai-0.3.0-py3-none-any.whl
```

---

## ✅ Verify Installation

```bash
# Test import
python -c "from clarify import enhance; print('clarify-ai ready!')"

# Test CLI
clarify --help

# Quick enhancement test
python -c "
from clarify import enhance
import cv2, numpy as np
img = np.zeros((64,64,3), dtype=np.uint8)
cv2.imwrite('/tmp/test.jpg', img)
print(enhance('/tmp/test.jpg', mode='classical'))
"
```

### Verify Hardware Acceleration

```bash
# Check ONNX Runtime providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Expected outputs:
# Linux/NVIDIA: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# macOS/Apple Silicon: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
# Others: ['CPUExecutionProvider']
```

---

## 🖼️ Usage

### 1. Basic Image Enhancement (AI, 2x)

```python
from clarify import enhance

enhanced_path = enhance("photo.jpg")
print(f"Saved to: {enhanced_path}")
# Output: photo_ai.png (2x upscaled + clarity polished)
```

### 2. Custom Upscaling (AI Mode)

```python
from clarify import enhance

# 2x upscaling (default)
enhance("photo.jpg", mode="ai", scale=2)

# 3x upscaling
enhance("photo.jpg", mode="ai", scale=3)

# 4x upscaling
enhance("photo.jpg", mode="ai", scale=4)
```

### 3. Classical Enhancement (Non-AI)

```python
from clarify import enhance

# No upscaling — denoising + detail enhancement + clarity polish
enhance("photo.jpg", mode="classical")
# Output: same dimensions, enhanced quality
```

### 4. Denoise (Classical Non-Local Means + Bilateral)

Removes Gaussian and color-sensor noise while retaining sharp edges.

```python
from clarify import enhance

# Default strength
enhance("noisy.jpg", mode="denoise")

# Custom strength (higher = stronger denoising)
enhance("noisy.jpg", mode="denoise", strength=1.5)
```

### 5. Deblur (LAB-space Unsharp Masking)

Recovers edges degraded by slight motion or optical blur.

```python
from clarify import enhance

# Default strength
enhance("blurry.jpg", mode="deblur")

# Custom strength (higher = stronger sharpening)
enhance("blurry.jpg", mode="deblur", strength=1.2)
```

### 6. Face Restore (GFPGAN ONNX)

Detects faces and restores them through a bundled GFPGAN ONNX model. Falls back to classical enhancement when no face is detected.

```python
from clarify import enhance

# Face restoration (requires gfpgan_v1_4.onnx bundled)
enhance("portrait.jpg", mode="face_restore")

# Output to high-quality WebP
enhance("portrait.jpg", mode="face_restore", output_format="webp", quality=95)
```

**Note:** Face restore requires the bundled `gfpgan_v1_4.onnx` model in `src/clarify/models/`. Generate it offline with `tools/export_gfpgan_onnx.py` (PyTorch only at build time, not runtime).

### 7. Multi-Format Output

Save in PNG, JPEG, WebP, or TIFF with configurable quality.

```python
from clarify import enhance

# JPEG with quality 92
enhance("noisy.jpg", mode="denoise", output_format="jpeg", quality=92)

# WebP compression
enhance("lowres.png", mode="ai", scale=4, output_format="webp", quality=90)

# TIFF (LZW compressed)
enhance("photo.jpg", mode="classical", output_format="tiff")

# "jpg" alias accepted for JPEG
enhance("photo.jpg", mode="classical", output_format="jpg", quality=85)
```

### 8. Custom Output Location

```python
from clarify import enhance

# Relative path (resolves next to input)
enhance("assets/input.jpg", scale=4, output="exports/ultra_res.png")

# Absolute path
enhance("photo.jpg", output="/home/user/output/enhanced.png")

# pathlib.Path objects work too
from pathlib import Path
enhance(Path("photo.jpg"), output=Path("results/enhanced.png"))
```

### 9. Batch Processing

```python
from clarify import enhance

images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]

for img in images:
    result = enhance(img, mode="ai", scale=2)
    print(f"Enhanced: {result}")

# Model cached after first call — subsequent calls are faster
```

---

## 💻 Command-Line Interface

The `clarify` command is available after `pip install -e .` or when running with `PYTHONPATH=src`.

```bash
# Basic AI enhancement (2x, PNG output)
clarify photo.jpg

# 4x AI upscaling
clarify photo.jpg --scale 4

# Classical enhancement (no upscaling)
clarify photo.jpg --mode classical

# Denoise an image to JPEG
clarify noisy.jpg --mode denoise --format jpeg --quality 90

# Deblur to TIFF
clarify blurry.jpg --mode deblur --strength 1.2 -f tiff

# Face restore to high-quality WebP
clarify portrait.png --mode face_restore -f webp -q 95

# Custom output
clarify photo.jpg --scale 2 --output exports/enhanced.png

# Help
clarify --help
```

### CLI Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `image` | — | `str` | Required | Target input image path |
| `--mode` | `-m` | `ai` \| `classical` \| `denoise` \| `deblur` \| `face_restore` | `ai` | Enhancement pipeline |
| `--scale` | `-s` | `2` \| `3` \| `4` | `2` | Upscaling factor (AI only) |
| `--format` | `-f` | `png` \| `jpeg` \| `webp` \| `tiff` | `png` | Output container format |
| `--quality` | `-q` | `1`–`100` | `95` | Encoding quality (JPEG/WebP) |
| `--strength` | — | `float` | `1.0` | Filter strength (denoise/deblur) |
| `--output` | `-o` | `str` | `None` | Custom output path |
| `--help` | `-h` | flag | — | Show help |

---

## 📚 API Reference

### `clarify.enhance()`

```python
def enhance(
    image: Union[str, Path],
    *,
    mode: Literal["ai", "classical", "denoise", "deblur", "face_restore"] = "ai",
    scale: Literal[2, 3, 4] = 2,
    output: Optional[Union[str, Path]] = None,
    output_format: Literal["png", "jpeg", "jpg", "webp", "tiff", "tif"] = "png",
    quality: int = 95,
    strength: float = 1.0,
) -> Path:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str \| Path` | Required | Input image path |
| `mode` | `Literal[...]` | `"ai"` | Enhancement pipeline |
| `scale` | `Literal[2, 3, 4]` | `2` | Upscaling factor (AI only) |
| `output` | `str \| Path \| None` | `None` | Custom output destination |
| `output_format` | `Literal[...]` | `"png"` | Output format: png, jpeg, webp, tiff |
| `quality` | `int` | `95` | Compression quality (1–100) for JPEG/WebP |
| `strength` | `float` | `1.0` | Filter strength for denoise/deblur |

#### Return Value

`pathlib.Path` — Absolute path to the enhanced image.

#### Default Output Naming

| Input | mode | Output |
|-------|------|--------|
| `input.jpg` | `ai` | `input_ai.png` |
| `input.jpeg` | `classical` | `input_classical.png` |
| `input.png` | `denoise` | `input_denoise.jpg` |
| `landscape.webp` | `deblur` | `landscape_deblur.webp` |
| `portrait.png` | `face_restore` | `portrait_face_restore.tiff` |

The output suffix matches `output_format` (`.jpg` for `jpeg`).

#### Exceptions

| Exception | Inherits From | When Raised |
|-----------|---------------|-------------|
| `ImageNotFoundError` | `FileNotFoundError` | Input image doesn't exist |
| `UnsupportedImageError` | `ClarifyError` | File can't be decoded |
| `ModelNotFoundError` | `FileNotFoundError` | Bundled model missing (e.g. face_restore without GFPGAN) |
| `ConfigurationError` | `ValueError` | Invalid mode, scale, or output format |

---

## 🗂️ Supported Image Formats

| Format | Extensions | Input | Output |
|--------|------------|-------|--------|
| JPEG | `.jpg`, `.jpeg` | ✅ | ✅ (quality configurable) |
| PNG | `.png` | ✅ | ✅ (lossless, compression level 3) |
| WebP | `.webp` | ✅ | ✅ (quality configurable) |
| TIFF | `.tiff`, `.tif` | ✅ | ✅ (LZW compressed) |
| Bitmap | `.bmp` | ✅ | ❌ |

---

## 🎨 Clarity Polish Pipeline

Every enhancement mode applies a **clarity polish** stage that transforms a "before"-state image into a polished, "after"-state result — sharper, higher contrast, and more vibrant, without over-processing or halo artifacts.

The polish runs **three stages** on every output:

| Stage | Technique | AI Mode Value | Effect |
|-------|-----------|---------------|--------|
| Contrast lift | LAB gamma curve | `contrast = 1.0 + 0.075 × strength` (→ **1.15**) | Brighter, more balanced tones |
| Edge definition | Unsharp mask | `alpha = 1.0 + 0.35 × strength` (→ **1.7**) | Maximum clean sharpness, no halos |
| Color vibrancy | HSV saturation boost | `sat = 1.0 + 0.075 × strength` (→ **1.15**) | Richer, more lifelike colors |

Parameters are pushed to the **greatest clean values** that remain artifact-free on portrait and beauty imagery — verified against before/after reference pairs.

```python
from clarify import enhance

# Before -> After quality (polished, print-ready)
enhance("portrait_before.jpg", mode="ai", scale=2, output_format="png")
# Sharpness: ~369 → ~738 | Contrast: ~46 → ~53 | Saturation: ~110 → ~115
```

The `strength` parameter scales all three polish stages proportionally:
- **AI mode**: fixed at `strength=2.0` for maximum clean polish
- **Classical / Denoise / Face Restore**: `strength=0.4` (lighter, these modes already sharpen)
- **Deblur**: uses the user-provided `--strength` (default `1.0`)

---

## ⚡ Hardware Acceleration

`clarify-ai` automatically selects the best available inference backend:

| Hardware | Backend | Priority |
|----------|---------|----------|
| NVIDIA GPU | CUDA | 1 |
| Apple Silicon | CoreML | 2 |
| CPU (any) | CPU | 3 |

**No configuration needed** — the library detects and uses the best available provider automatically.

---

## 🧠 AI Model

**EDSR (Enhanced Deep Residual Networks)** — a deep learning architecture for single-image super-resolution.

### Bundled Models (in `src/clarify/models/`)

| Model | Scale | Size |
|-------|-------|------|
| `edsr_x2.onnx` | 2x | ~5 MB |
| `edsr_x3.onnx` | 3x | ~6 MB |
| `edsr_x4.onnx` | 4x | ~6 MB |

**Source:** [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base) (exported via `tools/export_edsr_onnx.py`)

### Face Restore Model (build-time export)

| Model | Purpose | Size |
|-------|---------|------|
| `gfpgan_v1_4.onnx` | Face restoration (ONNX) | ~25 MB |

Generated offline via `tools/export_gfpgan_onnx.py` (PyTorch required only at build time, **not** at runtime).

### Processing Pipeline (AI Mode)

```text
Input Image → Path Resolution → Hardware Selector → EDSR ONNX Inference
       → Clarity Polish (contrast + unsharp + saturation) → Multi-format Encode → Output Path
```

1. Load input image → BGR numpy array
2. Convert to RGB float32 tensor (normalized to [0,1])
3. Run ONNX Runtime inference (auto-selected provider)
4. Clip & convert output tensor → BGR uint8
5. **Clarity polish**: LAB gamma contrast + unsharp mask + HSV saturation boost
6. Encode to target format → save to disk
7. Return `pathlib.Path` to output

### Processing Pipeline (Face Restore Mode)

1. Load input image → BGR numpy array
2. Detect faces via Haar cascade (OpenCV built-in)
3. Crop each face with 20% margin, resize to 512×512
4. Normalize to [-1, 1], run GFPGAN ONNX inference
5. Denormalize, resize back to original crop size
6. Paste restored faces into result image
7. **Clarity polish**: LAB gamma contrast + unsharp mask + HSV saturation boost
8. Encode to target format → save to disk

---

## 🏗️ Architecture

```text
┌──────────────────────┐
│   Input Image Path   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Path Resolution    │  ← Caller-aware (inspect.stack)
│   (relative to       │
│    calling script)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Hardware Selector  │  ← CUDA → CoreML → CPU
│   (ONNX providers)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Mode Dispatcher    │  ← ai / classical / denoise /
│                      │    deblur / face_restore
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Model Cache        │  ← Per-scale ONNX sessions
│   (EDSR ONNX)        │    (GFPGAN ONNX for face_restore)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Model Inference    │  ← ONNX Runtime / OpenCV
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Image Reconstruction │  ← Tensor → BGR numpy
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Clarity Polish     │  ← LAB contrast + unsharp + HSV sat
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Disk Serializer    │  ← PNG / JPEG / WebP / TIFF
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    pathlib.Path      │  ← Return value
└──────────────────────┘
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Path Resolution | `core.py:_resolve_path()` | Resolves relative paths against caller's directory |
| Hardware Selector | `core.py:_get_edsr_session()` | Picks CUDA/CoreML/CPU provider |
| Model Cache | `core.py:_CACHED_SESSIONS` | In-memory per-scale session cache |
| AI Inference | `core.py:_enhance_ai()` | ONNX Runtime EDSR inference |
| Classical Enhance | `core.py:_enhance_classical()` | OpenCV denoise + detail enhance |
| Denoise | `core.py:_enhance_denoise()` | NLMeans + bilateral filtering |
| Deblur | `core.py:_enhance_deblur()` | LAB-space unsharp masking |
| Face Restore | `core.py:_enhance_face_restore()` | GFPGAN ONNX + face detection |
| Clarity Polish | `core.py:_polish()` | LAB contrast + unsharp + HSV saturation |
| Serialization | `core.py:_encode_and_save()` | Multi-format encoding (PNG/JPEG/WebP/TIFF) |
| CLI | `cli.py:main()` | argparse-based terminal interface |
| Exceptions | `exceptions.py` | ClarifyError hierarchy + ConfigurationError |

---

## 🔧 Development

### Setup

```bash
git clone https://github.com/your-username/clarify.git
cd clarify

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e ".[dev]"
```

### Commands

```bash
# Run tests
pytest -v

# Type check (strict)
mypy src/clarify --strict

# Format code
black .

# Export EDSR ONNX models (requires PyTorch + super-image)
python tools/export_edsr_onnx.py

# Export GFPGAN ONNX model (requires PyTorch + gfpgan, build-time only)
python tools/export_gfpgan_onnx.py

# Benchmark
python tools/benchmark.py
```

### Project Structure

```
clarify/
├── src/clarify/           # Main package
│   ├── __init__.py        # Public exports
│   ├── core.py            # Enhancement logic
│   ├── cli.py             # CLI entry point
│   ├── exceptions.py      # Custom exceptions
│   ├── py.typed           # PEP 561 marker
│   └── models/            # Bundled ONNX models
│       ├── edsr_x2.onnx
│       ├── edsr_x3.onnx
│       ├── edsr_x4.onnx
│       └── gfpgan_v1_4.onnx  (optional, face_restore)
├── tests/                 # Test suite
├── tools/                 # Development scripts
│   ├── export_edsr_onnx.py
│   └── export_gfpgan_onnx.py
├── pyproject.toml         # Build config (hatchling)
└── README.md
```

---

## 🧪 Testing

```bash
# Full test suite
pytest -v

# With coverage
pytest --cov=clarify --cov-report=term-missing
```

**Test coverage includes:**
- AI mode: 2x, 3x, 4x upscaling with dimension verification
- Classical mode: dimension preservation
- Denoise: dimension preservation, strength bounds
- Deblur: dimension preservation, strength bounds
- Face restore: missing model error handling
- Multi-format output: PNG, JPEG, WebP, TIFF suffixes
- Quality clamping: out-of-range values clamped to 1–100
- Error handling: missing files, invalid formats, bad parameters
- Input formats: JPEG, PNG, WebP
- Path resolution: absolute, relative, custom output
- Real-time example: `test_real_example.py` with `input.jpeg`

---

## 📦 Distributing to Others

### For Team Members / CI

```bash
# Option 1: Shared wheel (build once, distribute .whl)
python -m build
# Share dist/clarify_ai-0.3.0-py3-none-any.whl
pip install clarify_ai-0.3.0-py3-none-any.whl

# Option 2: Git dependency (pip install from repo)
pip install git+https://github.com/your-username/clarify.git

# Option 3: Git submodule
git submodule add https://github.com/your-username/clarify.git vendor/clarify
# Then: pip install -e vendor/clarify
```

### For End Users (Non-developers)

1. **Pre-built wheel** — Send them the `.whl` file, they run `pip install clarify_ai-0.3.0-py3-none-any.whl`
2. **Standalone script** — Vendor `src/clarify` into your app, no install needed
3. **Docker** — Create a container with the package pre-installed

---

## 🛣️ Roadmap

- [ ] Additional SOTA super-resolution architectures (Real-ESRGAN, SwinIR)
- [ ] Batch/directory processing
- [ ] Async enhancement support
- [ ] Progress callbacks for large images
- [ ] Configurable model selection
- [ ] Benchmark suite with comparison baselines

---

## 🤝 Contributing

```bash
# 1. Fork & clone
git clone https://github.com/your-fork/clarify.git
cd clarify

# 2. Create feature branch
git checkout -b feature/my-improvement

# 3. Develop
# ... make changes ...

# 4. Validate
pytest -v
mypy src/clarify --strict
black --check .

# 5. Submit PR
git push origin feature/my-improvement
# Open pull request
```

For major changes, open an issue first to discuss.

---

## 📄 License

**MIT License** — see [LICENSE](LICENSE) for details.

Free for personal and commercial use.

---

## ❤️ Acknowledgements

- [ONNX Runtime](https://onnxruntime.ai/) — High-performance inference engine
- [OpenCV](https://opencv.org/) — Classical image processing
- [EDSR](https://huggingface.co/eugenesiow/edsr-base) — Enhanced Deep Residual Networks
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face restoration (ONNX export)
- [super-image](https://github.com/eugenesiow/super-image) — PyTorch EDSR implementation
- [Hugging Face](https://huggingface.co/) — Model hosting
- [Pillow](https://python-pillow.org/) — Image I/O support

---

<div align="center">

### clarify-ai

**Making AI-powered image enhancement as simple as cloning a repo.**

No PyPI. No internet. No hassle. No heavy ML frameworks at runtime. Maximum clean clarity on every output.

Made for designers and developers who want it to just work.

</div>
