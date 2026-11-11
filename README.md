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
</p>

**Enhance your images with just one function call.**

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

**clarify-ai abstracts this entire workflow into a simple Python interface** — bundled with ONNX models, zero external dependencies at runtime.

```python
from clarify import enhance

enhanced_path = enhance("photo.jpg")
print(f"Enhanced image saved at: {enhanced_path}")
```

By default, the library:
1. Resolves the input image path (relative to your script)
2. Automatically selects the best available hardware (CUDA → CoreML → CPU)
3. Loads the bundled EDSR super-resolution model
4. Performs AI-powered image upscaling
5. Exports the enhanced image as lossless PNG
6. Returns the absolute output path as a `pathlib.Path` object

**Example project structure:**
```text
my-project/
├── main.py          # your script
├── photo.jpg        # input image
└── photo_enhanced.png  # generated output
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Zero-config API** | Enhance images with a single function call |
| **AI Super-Resolution** | Pretrained EDSR (Enhanced Deep Residual Networks) bundled |
| **Classical Enhancement** | OpenCV-based denoising + detail enhancement (no upscaling) |
| **Auto Hardware Detection** | CUDA (NVIDIA) → CoreML (Apple Silicon) → CPU fallback |
| **In-Memory Model Caching** | Reuses loaded models across calls |
| **Smart Path Resolution** | Relative paths resolve against calling script's directory |
| **Multiple Upscaling Factors** | 2x, 3x, 4x (AI mode only) |
| **Lossless Output** | PNG format by default |
| **Custom Output Paths** | Specify any destination |
| **CLI Included** | `clarify` terminal command |
| **Fully Typed** | `py.typed` + mypy strict compliance |
| **No Runtime Downloads** | All models bundled (~17 MB) |

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
pip install -r requirements.txt  # or: pip install numpy opencv-python-headless onnxruntime pillow

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
python -m build  # Creates dist/clarify_ai-0.2.0-py3-none-any.whl

# Copy .whl to target machine and install
pip install clarify_ai-0.2.0-py3-none-any.whl
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
# Output: photo_enhanced.png (2x upscaled)
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

# No upscaling — denoising + detail enhancement only
enhance("photo.jpg", mode="classical")
# Output: same dimensions, enhanced quality
```

### 4. Custom Output Location

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

### 5. Batch Processing

```python
from clarify import enhance

images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]

for img in images:
    result = enhance(img, mode="ai", scale=2)
    print(f"Enhanced: {result}")

# Model cached after first call — subsequent calls are faster
```

### 6. Using pathlib.Path

```python
from pathlib import Path
from clarify import enhance

image_path = Path("assets/photo.jpg")
enhanced_path = enhance(image_path, scale=3)
print(enhanced_path)  # PosixPath('/full/path/to/photo_enhanced.png')
```

---

## 💻 Command-Line Interface

The `clarify` command is available after `pip install -e .` or when running with `PYTHONPATH=src`.

```bash
# Basic AI enhancement (2x)
clarify photo.jpg

# 4x AI upscaling
clarify photo.jpg --scale 4

# Classical enhancement (no upscaling)
clarify photo.jpg --mode classical

# Custom output
clarify photo.jpg --scale 2 --output exports/enhanced.png

# Help
clarify --help
```

### CLI Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--mode` | `-m` | `ai` \| `classical` | `ai` | Enhancement pipeline |
| `--scale` | `-s` | `2` \| `3` \| `4` | `2` | Upscaling factor (AI only) |
| `--output` | `-o` | `str` | `None` | Custom output path |
| `--help` | `-h` | flag | — | Show help |

---

## 📚 API Reference

### `clarify.enhance()`

```python
def enhance(
    image: Union[str, Path],
    *,
    mode: Literal["ai", "classical"] = "ai",
    scale: Literal[2, 3, 4] = 2,
    output: Optional[Union[str, Path]] = None,
) -> Path:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str \| Path` | Required | Input image path |
| `mode` | `Literal["ai", "classical"]` | `"ai"` | Enhancement pipeline |
| `scale` | `Literal[2, 3, 4]` | `2` | Upscaling factor (AI only) |
| `output` | `str \| Path \| None` | `None` | Custom output destination |

#### Return Value

`pathlib.Path` — Absolute path to the enhanced image.

#### Default Output Naming

| Input | Generated Output |
|-------|------------------|
| `input.jpg` | `input_enhanced.png` |
| `input.jpeg` | `input_enhanced.png` |
| `input.png` | `input_enhanced.png` |
| `landscape.webp` | `landscape_enhanced.png` |
| `image.bmp` | `image_enhanced.png` |

#### Exceptions

| Exception | Inherits From | When Raised |
|-----------|---------------|-------------|
| `ImageNotFoundError` | `FileNotFoundError` | Input image doesn't exist |
| `UnsupportedImageError` | `ClarifyError` | File can't be decoded |
| `ModelNotFoundError` | `FileNotFoundError` | Bundled model missing |
| `ValueError` | — | Invalid `mode` or `scale` |

---

## 🗂️ Supported Image Formats

| Format | Extensions | Input |
|--------|------------|-------|
| JPEG | `.jpg`, `.jpeg` | ✅ |
| PNG | `.png` | ✅ |
| WebP | `.webp` | ✅ |
| Bitmap | `.bmp` | ✅ |

**Output:** Always PNG (lossless). PNG avoids additional lossy compression but cannot recover information absent from the original.

---

## ⚡ Hardware Acceleration

`clarify-ai` automatically selects the best available inference backend:

| Hardware | Backend | Priority |
|----------|---------|----------|
| NVIDIA GPU | CUDA | 1 |
| Apple Silicon | CoreML | 2 |
| CPU (any) | CPU | 3 |

### Device Selection Logic

```text
              ┌──────────────────┐
              │      Start       │
              └────────┬─────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │ Check CUDA Available? │
           └───────────┬───────────┘
                       │
                  ┌────┴────┐
                  ▼         ▼
               Yes        No
                  │         │
                  ▼         ▼
           ┌─────────┐ ┌──────────────────┐
           │ CUDA GPU│ │ Check CoreML?    │
           └─────────┘ └────────┬─────────┘
                                │
                           ┌────┴────┐
                           ▼         ▼
                          Yes        No
                           │         │
                           ▼         ▼
                    ┌──────────┐ ┌─────────┐
                    │  CoreML  │ │   CPU   │
                    │ (Apple)  │ │ Fallback│
                    └──────────┘ └─────────┘
```

**No configuration needed** — the library detects and uses the best available provider automatically.

---

## 🧠 AI Model

**EDSR (Enhanced Deep Residual Networks)** — a deep learning architecture for single-image super-resolution.

### Bundled Models (in `src/clarify/models/`)

| Model | Scale | Size |
|-------|-------|------|
| `edsr_x2.onnx` | 2x | 5.2 MB |
| `edsr_x3.onnx` | 3x | 5.9 MB |
| `edsr_x4.onnx` | 4x | 5.8 MB |

**Source:** [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base) (exported via `tools/export_edsr_onnx.py`)

### Processing Pipeline

1. Load input image → BGR numpy array
2. Convert to RGB float32 tensor (normalized to [0,1])
3. Run ONNX Runtime inference (auto-selected provider)
4. Clip & convert output tensor → BGR uint8
5. Encode as PNG → save to disk
6. Return `pathlib.Path` to output

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
│  Hardware Selector   │  ← CUDA → CoreML → CPU
│   (ONNX providers)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Model Cache        │  ← Per-scale singleton sessions
│   (EDSR ONNX)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Model Inference    │  ← EDSR super-resolution
│   (ONNX Runtime)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Image Reconstruction │  ← Tensor → BGR numpy
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Disk Serializer    │  ← cv2.imencode → PNG
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
| Hardware Selector | `core.py:_get_onnx_session()` | Picks CUDA/CoreML/CPU provider |
| Model Cache | `core.py:_CACHED_SESSIONS` | In-memory per-scale session cache |
| AI Inference | `core.py:_enhance_ai()` | ONNX Runtime EDSR inference |
| Classical Enhance | `core.py:_enhance_classical()` | OpenCV denoise + detail enhance |
| Serialization | `core.py:enhance()` | PNG encoding + path return |

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

# Export ONNX models (requires PyTorch + super-image)
python tools/export_edsr_onnx.py

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
│       └── edsr_x4.onnx
├── tests/                 # Test suite
├── tools/                 # Development scripts
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
- Error handling: missing files, invalid formats, bad parameters
- Input formats: JPEG, PNG, WebP
- Path resolution: absolute, relative, custom output
- CLI integration

---

## 📦 Distributing to Others

### For Team Members / CI

```bash
# Option 1: Shared wheel (build once, distribute .whl)
python -m build
# Share dist/clarify_ai-0.2.0-py3-none-any.whl
pip install clarify_ai-0.2.0-py3-none-any.whl

# Option 2: Git dependency (pip install from repo)
pip install git+https://github.com/your-username/clarify.git

# Option 3: Git submodule
git submodule add https://github.com/your-username/clarify.git vendor/clarify
# Then: pip install -e vendor/clarify
```

### For End Users (Non-developers)

1. **Pre-built wheel** — Send them the `.whl` file, they run `pip install clarify_ai-0.2.0-py3-none-any.whl`
2. **Standalone script** — Vendor `src/clarify` into your app, no install needed
3. **Docker** — Create a container with the package pre-installed

---

## 🛣️ Roadmap

- [ ] Additional SOTA super-resolution architectures (Real-ESRGAN, SwinIR)
- [ ] AI-powered image denoising
- [ ] Face restoration (GFPGAN/CodeFormer)
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
- [super-image](https://github.com/eugenesiow/super-image) — PyTorch EDSR implementation
- [Hugging Face](https://huggingface.co/) — Model hosting
- [Pillow](https://python-pillow.org/) — Image I/O support

---

<div align="center">

### clarify-ai

**Making AI-powered image enhancement as simple as cloning a repo.**

No PyPI. No internet. No hassle.

Made for developers who want it to just work.

</div>