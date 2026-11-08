<div align="center">

# clarify-ai

**Production-grade, zero-config AI image upscaler and enhancement engine for Python.**

<p align="center">
  <a href="https://pypi.org/project/clarify-ai/">
    <img src="https://img.shields.io/pypi/v/clarify-ai?color=blue&style=flat-square" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/clarify-ai/">
    <img src="https://img.shields.io/pypi/pyversions/clarify-ai?style=flat-square" alt="Python Versions">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="Black">
  <a href="https://mypy-lang.org/">
    <img src="https://img.shields.io/badge/mypy-checked-blue.svg?style=flat-square" alt="Mypy">
  </a>
</p>

<!-- <p align="center">
  <img src="https://img.shields.io/badge/PyPI-Not%20Published-orange?style=flat-square" alt="PyPI Status">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square" alt="Black">
  <img src="https://img.shields.io/badge/Type%20Checking-Mypy-blue?style=flat-square" alt="Mypy">
</p> -->

**Enhance your images with just one function call.**

No manual model configuration. No tensor manipulation. No complicated image-processing pipelines.

[Features](#-key-features) •
[Installation](#-installation) •
[Quickstart](#-quickstart) •
[CLI](#-command-line-interface) •
[API Reference](#-api-reference) •
[Architecture](#-architecture) •
[Contributing](#-contributing)

</div>

---

## 📌 Overview

`clarify-ai` is a Python image enhancement and super-resolution library designed to make AI-powered image upscaling as simple as downloading videos with `yt-dlp`.

Traditionally, implementing AI-based image enhancement requires developers to:

- Configure deep learning frameworks.
- Download and manage pretrained model weights.
- Handle image-to-tensor conversions.
- Configure CUDA or other hardware accelerators.
- Manage device transfers and inference.
- Reconstruct enhanced images.
- Handle image encoding and output paths.

**clarify-ai abstracts this entire workflow into a simple Python interface.**

Simply provide the path to an image, choose an optional upscaling factor, and let the library handle the enhancement pipeline.

```python
from clarify import enhance

enhanced_path = enhance("photo.jpg")

print(f"Enhanced image saved at: {enhanced_path}")
```

By default, the library:

1. Resolves the input image path.
2. Automatically selects the available hardware.
3. Loads the pretrained EDSR super-resolution model.
4. Performs AI-powered image upscaling.
5. Exports the enhanced image as a PNG.
6. Returns the absolute output path as a `pathlib.Path` object.

The enhanced image is automatically saved alongside the original image.

**Example:**

```text
project/
├── main.py
├── photo.jpg
└── photo_enhanced.png
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| Minimal Python API | Enhance images with a single function call. |
| AI Super-Resolution | Uses pretrained EDSR (Enhanced Deep Residual Networks). |
| Automatic Hardware Detection | Detects NVIDIA CUDA, Apple Silicon MPS, or CPU. |
| Intelligent Device Routing | Automatically selects an available inference device. |
| In-Memory Model Caching | Reuses loaded model weights across subsequent enhancement calls. |
| Automatic Path Resolution | Resolves relative paths against the calling script's directory. |
| Multiple Upscaling Factors | Supports 2x, 3x, and 4x upscaling. |
| Lossless Output Format | Saves enhanced images as PNG by default. |
| Custom Output Paths | Allows developers to specify their preferred output location. |
| Command-Line Interface | Provides a `clarify` terminal command. |
| Pythonic Return Values | Returns the generated image path as a `pathlib.Path` object. |

---

## 📦 Installation

Install the package using pip:

```bash
pip install clarify-ai
```

### Verify Installation

```bash
python -c "from clarify import enhance; print('clarify-ai installed successfully!')"
```

### Verify Hardware Acceleration (Optional)

Check whether PyTorch can access NVIDIA CUDA:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

For Apple Silicon MPS:

```bash
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

---

## 🚀 Quickstart

### Basic Image Enhancement

```python
from clarify import enhance

# Automatically enhances the image using the default 2x scale
enhanced_path = enhance("photo.jpg")

print(f"File saved to: {enhanced_path}")
```

**Expected output:**

```text
File saved to: /path/to/your/project/photo_enhanced.png
```

The returned value is a `pathlib.Path` object pointing to the generated image.

---

## 🖼️ Usage

### 1. Basic Image Enhancement

```python
from clarify import enhance

enhance("photo.jpg")
```

Automatically:

- Detects the image.
- Resolves its location.
- Loads the enhancement model.
- Performs 2x super-resolution.
- Saves the output as `photo_enhanced.png`.

### 2. Custom Upscaling

Choose between 2x, 3x, and 4x upscaling:

```python
from clarify import enhance

# 2x upscaling
enhance("photo.jpg", scale=2)

# 3x upscaling
enhance("photo.jpg", scale=3)

# 4x upscaling
enhance("photo.jpg", scale=4)
```

### 3. Custom Output Location

Specify a custom output filename or destination:

```python
from clarify import enhance

output_file = enhance(
    "assets/input.jpg",
    scale=4,
    output="exports/ultra_res.png"
)

print(f"Enhanced image: {output_file}")
```

### 4. Using pathlib.Path

You can also pass `pathlib.Path` objects instead of strings:

```python
from pathlib import Path
from clarify import enhance

image_path = Path("assets/photo.jpg")

enhanced_path = enhance(
    image_path,
    scale=2
)

print(enhanced_path)
```

### 5. Repeated Enhancement

The model is cached in memory so subsequent calls can reuse the initialized model.

```python
from clarify import enhance

images = [
    "photo1.jpg",
    "photo2.jpg",
    "photo3.jpg"
]

for image in images:
    result = enhance(image, scale=2)
    print(f"Enhanced: {result}")
```

---

## 💻 Command-Line Interface

`clarify-ai` includes a command-line executable called `clarify`.

Once installed, you can enhance images directly from your terminal without writing Python scripts.

### Basic Enhancement

```bash
clarify photo.jpg
```

### 4x Upscaling

```bash
clarify photo.jpg --scale 4
```

### Custom Output Destination

```bash
clarify photo.jpg --scale 2 --output exports/enhanced.png
```

### Display Help

```bash
clarify --help
```

### CLI Arguments

| Argument | Short | Type | Default | Description |
|---|---|---|---|---|
| `--scale` | `-s` | Integer | `2` | Upscaling factor: 2, 3, or 4. |
| `--output` | `-o` | String | `None` | Custom output filename or path. |
| `--help` | `-h` | Flag | — | Displays command-line help. |

---

## 📚 API Reference

### `clarify.enhance()`

The primary public interface for image enhancement.

```python
def enhance(
    image: Union[str, Path],
    *,
    scale: int = 2,
    output: Optional[Union[str, Path]] = None,
) -> Path:
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str \| Path` | Required | Input image path. |
| `scale` | `int` | `2` | Upscaling factor. Accepted values: 2, 3, or 4. |
| `output` | `str \| Path \| None` | `None` | Custom output destination. |

### Return Value

```python
pathlib.Path
```

Returns the fully resolved absolute path of the enhanced image.

### Default Output Naming

If no output destination is specified:

| Input | Generated Output |
|---|---|
| `input.jpg` | `input_enhanced.png` |
| `input.jpeg` | `input_enhanced.png` |
| `input.png` | `input_enhanced.png` |
| `landscape.webp` | `landscape_enhanced.png` |
| `image.bmp` | `image_enhanced.png` |

### Exceptions

| Exception | Description |
|---|---|
| `ImageNotFoundError` | Raised when the specified input image does not exist. |
| `UnsupportedImageError` | Raised when the input image cannot be decoded or its format is unsupported. |

---

## 🗂️ Supported Image Formats

| Format | Extension | Input Support |
|---|---|---|
| JPEG | `.jpg`, `.jpeg` | ✅ |
| PNG | `.png` | ✅ |
| WebP | `.webp` | ✅ |
| Bitmap | `.bmp` | ✅ |

**Default output format:** PNG.

PNG encoding avoids additional lossy JPEG compression during output serialization. However, lossless output does not recover image information that was absent from the original image.

---

## ⚡ Hardware Acceleration

`clarify-ai` automatically selects an available inference device.

| Hardware | Backend | Priority |
|---|---|---|
| NVIDIA GPU | CUDA | 1 |
| Apple Silicon | MPS | 2 |
| Processor | CPU | 3 |

### Device Selection Logic

```text
             ┌──────────────────┐
             │      Start       │
             └────────┬─────────┘
                      |
                      v
          ┌───────────────────────┐
          │ Check CUDA Availability│
          └───────────┬───────────┘
                      |
                 +----+----+
                 |         |
                Yes        No
                 |         |
                 v         v
           ┌─────────┐  ┌─────────────┐
           │ CUDA GPU│  │ Check MPS   │
           └─────────┘  └──────┬──────┘
                               |
                          +----+----+
                          |         |
                         Yes        No
                          |         |
                          v         v
                    ┌─────────┐ ┌─────────┐
                    │ Apple   │ │   CPU   │
                    │   MPS   │ │ Fallback│
                    └─────────┘ └─────────┘
```

The library automatically falls back to CPU execution when a supported GPU backend is unavailable.

---

## 🧠 AI Model

The library uses **EDSR (Enhanced Deep Residual Networks)**, a deep learning architecture designed for single-image super-resolution.

### Pretrained Model

[eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base)

### Model Processing Pipeline

1. Load the input image.
2. Convert the image into the required tensor representation.
3. Transfer the tensor to the selected inference device.
4. Perform super-resolution using the pretrained EDSR network.
5. Reconstruct the enhanced image.
6. Serialize the result as a PNG file.

The pretrained model is cached in memory to reduce repeated initialization overhead.

---

## 🏗️ Architecture

The internal processing pipeline follows a modular architecture:

```text
                 ┌──────────────────────┐
                 │    Input Image Path  │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │   Path Resolution    │
                 │  Caller Context      │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │  Hardware Selector   │
                 │   CUDA / MPS / CPU   │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │   Singleton Cache    │
                 │  EDSR Model Weights  │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │   Model Inference    │
                 │  Super-Resolution    │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │ Image Reconstruction│
                 │   Tensor → Image     │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │   Disk Serializer    │
                 │     PNG Output       │
                 └──────────┬───────────┘
                            |
                            v
                 ┌──────────────────────┐
                 │    pathlib.Path      │
                 │    Return Value      │
                 └──────────────────────┘
```

### Core Components

**1. Path Resolution**

Determines the appropriate base directory for relative input and output paths using the calling script's location.

**2. Hardware Selector**

Detects supported hardware acceleration backends and selects the appropriate inference device.

**3. Model Cache**

Maintains an in-memory model instance to reduce repeated initialization overhead.

**4. Model Inference**

Processes image tensors through the pretrained EDSR super-resolution architecture.

**5. Image Reconstruction**

Converts the model output into an image suitable for serialization.

**6. Disk Serializer**

Exports the enhanced image in PNG format and returns its destination path.

---

## 🔧 Development

Contributions, bug reports, feature suggestions, and improvements are welcome.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/clarify.git

cd clarify
```

Replace `your-username` with your actual GitHub username.

### 2. Create a Virtual Environment

**Linux / macOS:**

```bash
python -m venv .venv

source .venv/bin/activate
```

**Windows:**

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Run the Test Suite

```bash
pytest -v
```

### 5. Run Code Formatting

```bash
black .
```

### 6. Run Static Type Checking

```bash
mypy clarify
```

---

## 🧪 Testing

Before submitting changes, run the automated test suite:

```bash
pytest -v
```

Recommended testing scenarios:

- Valid JPEG input.
- Valid PNG input.
- Valid WebP input.
- Valid BMP input.
- Missing input file.
- Unsupported image format.
- Invalid upscaling factor.
- Custom output destination.
- Relative input paths.
- Absolute input paths.
- CUDA inference.
- MPS inference.
- CPU fallback.
- Repeated calls using the model cache.
- CLI execution.

---

## 🛣️ Roadmap

The long-term vision of `clarify-ai` is to provide developers with a simple, reliable, and extensible Python interface for AI-powered image enhancement.

Planned areas of development include:

- [ ] Additional state-of-the-art super-resolution architectures.
- [ ] AI-powered image denoising.
- [ ] Image sharpening and deblurring.
- [ ] Face restoration and enhancement.
- [ ] Batch image processing.
- [ ] Directory-level image enhancement.
- [ ] Configurable model selection.
- [ ] Improved inference performance.
- [ ] Advanced GPU memory management.
- [ ] Support for additional image formats.
- [ ] Enhanced benchmarking and model evaluation.
- [ ] Progress reporting for large images.
- [ ] Async image enhancement support.

---

## 🤝 Contributing

Contributions are highly appreciated!

If you have an idea for improving `clarify-ai`, feel free to:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Add or update tests.
5. Run the test suite.
6. Submit a pull request.

For major changes, consider opening an issue first to discuss the proposed implementation.

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## ❤️ Acknowledgements

- [PyTorch](https://pytorch.org/) — Deep learning framework.
- [Hugging Face](https://huggingface.co/) — Pretrained model hosting and distribution.
- [EDSR](https://huggingface.co/eugenesiow/edsr-base) — Enhanced Deep Residual Networks for Image Super-Resolution.
- [Pillow](https://python-pillow.org/) — Image processing and manipulation.

---

<div align="center">

### clarify-ai

**Making AI-powered image enhancement as simple as importing a Python library.**

Made with ❤️ for the Python community.

</div>