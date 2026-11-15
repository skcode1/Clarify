# clarify-ai: Technical Architecture & Design Documentation

> **Zero-config AI image super-resolution via ONNX Runtime** — Production-grade, local-first, offline-ready. Pluggable model registry: SPAN (default), SRVGGNet Compact, RealPLSKR, EDSR Baseline — all sub-50 MB, zero PyTorch at runtime. Includes a Model Context Protocol (MCP) server for agent integration.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Technology Deep Dive](#core-technology-deep-dive)
   - [EDSR Model](#1-edrs-enhanced-deep-residual-networks)
   - [ONNX Runtime Inference Engine](#2-onnx-runtime-inference-engine)
   - [Classical Enhancement Pipeline](#3-classical-enhancement-pipeline-non-ai-mode)
   - [Smart Path Resolution](#4-smart-path-resolution)
4. [Tech Stack Breakdown](#tech-stack-breakdown)
5. [Packaging & Distribution Strategy](#packaging--distribution-strategy)
6. [Performance Characteristics](#performance-characteristics)
7. [Reliability & Correctness](#reliability--correctness)
8. [Deployment Scenarios](#deployment-scenarios)
9. [Extensibility Points](#extensibility-points)
10. [MCP Server](#mcp-server)
11. [Competitive Positioning](#competitive-positioning)
12. [Value Proposition](#value-proposition)
13. [Architecture Summary](#architecture-summary)

---

## Executive Summary

**clarify-ai** reduces the complexity of deploying production-grade AI image enhancement from weeks of ML engineering to a single function call.

```
Traditional Approach:          clarify-ai Approach:
┌─────────────────────┐        ┌─────────────────────┐
│ PyTorch/TF install  │        │ git clone + pip     │
│ Model download      │   →    │ (models bundled)    │
│ Device config       │        │ Auto CUDA/CoreML    │
│ Tensor pipelines    │        │ Single enhance()    │
│ Output handling     │        │ Returns pathlib.Path│
└─────────────────────┘        └─────────────────────┘
     Weeks of work                  Minutes to integrate
```

**Core Value Proposition:** *Enterprise-grade AI inference with zero operational overhead.*

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        clarify-ai PROCESSING PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐   │
│  │   INPUT      │    │  PATH RESOLUTION │    │  IMAGE DECODING    │   │
│  │  (str/Path)  │───▶│  (caller-aware)  │───▶│  (cv2.imdecode)    │   │
│  └──────────────┘    └──────────────────┘    └─────────┬──────────┘   │
│                                                         │              │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────▼──────────┐   │
│  │   OUTPUT     │    │  DISK SERIALIZER │    │  MODE SELECTION    │   │
│  │  (Path)      │◀───│  (cv2.imencode)  │◀───│  (AI / Classical)  │   │
│  └──────────────┘    └──────────────────┘    └─────────┬──────────┘   │
│                                                         │              │
│                    ┌────────────────────────────────────▼────────┐    │
│                    │           HARDWARE ABSTRACTION LAYER        │    │
│                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │    │
│                    │  │  CUDA    │ │ CoreML   │ │   CPU    │     │    │
│                    │  │(NVIDIA)  │ │(Apple)   │ │(Fallback)│     │    │
│                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘     │    │
│                    └─────────┼───────────┼───────────┼────────────┘    │
│                              │           │           │                 │
│                    ┌─────────▼───────────▼───────────▼────────┐       │
│                    │         ONNX RUNTIME EXECUTION ENGINE     │       │
│                    │  ┌────────────────────────────────────┐   │       │
│                    │  │      PLUGGABLE MODEL REGISTRY          │   │       │
│                    │  │  (SPAN / SRVGG Compact / RealPLSKR     │   │       │
│                    │  │   / EDSR Baseline ONNX presets)        │   │       │
│                    │  │  • Per-(arch, scale) session caching   │   │       │
│                    │  │  • Graph optimization (ORT_ENABLE_ALL) │   │       │
│                    │  │  • Custom model_path support           │   │       │
│                    │  └────────────────────────────────────┘   │       │
│                    └────────────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Technology Deep Dive

### 1. Pluggable Model Registry

`clarify-ai` uses a pluggable model registry (`core.py:MODEL_PRESETS`) that lets you select any ONNX super-resolution architecture via the `arch` parameter, or supply a custom `model_path`.

| Arch | Filename | Approx Size | Best For |
|------|----------|-------------|----------|
| `span` (default) | `span_x2.onnx` | ~1.7–5 MB | General-purpose photos, fast CPU pipelines |
| `sr_compact` | `realesr-general-x4v3.onnx` | ~5 MB | Web images, compressed JPEGs, illustrations |
| `realplskr` | `realplskr_x4.onnx` | ~30 MB | High-detail portraits, realistic photography (HQ tier) |
| `edsr_baseline` | `edsr_x2.onnx` | ~5–6 MB | Clean synthetic graphics, lossless renders (legacy) |

**Model Selection Order:**
1. `model_path` argument (if provided) → validate + use directly (air-gapped/custom)
2. `arch` preset → bundled model from `src/clarify/models/`
3. Default → `span` at 2× scale

**Session Cache:** `_CACHED_SESSIONS: dict[tuple[arch, scale], InferenceSession]` — sessions are keyed by architecture and scale, so different models for the same scale coexist without collision.

---

### 2. ONNX Runtime Inference Engine

```python
# Provider priority (auto-detected at first call)
providers = [
    "CUDAExecutionProvider",      # NVIDIA GPU (priority 1)
    "CoreMLExecutionProvider",    # Apple Silicon Neural Engine (priority 2)  
    "CPUExecutionProvider"        # Universal fallback (priority 3)
]

# Session options for maximum performance
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Enables: constant folding, operator fusion, layout optimization, dead code elimination
```

**Key Design Decisions:**
- **No PyTorch/TensorFlow runtime** — ONNX Runtime is ~50 MB vs 2+ GB
- **Dynamic axes** — Single model handles any input resolution (no tiling needed for typical images)
- **Per-(arch, scale) session caching** — `_CACHED_SESSIONS: dict[tuple[arch, scale], InferenceSession]` avoids reload on repeated calls; different architectures for the same scale coexist without collision
- **Graph optimization** — ORT_ENABLE_ALL fuses conv+bias+relu, optimizes memory layout

---

### 3. Classical Enhancement Pipeline (Non-AI Mode)

```python
def _enhance_classical(img_bgr: np.ndarray) -> np.ndarray:
    # 1. Denoise: Non-local means (preserves edges better than Gaussian)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_bgr, None, 
        h=5, hColor=5,          # Filter strength
        templateWindowSize=7,   # Patch size
        searchWindowSize=21     # Search window
    )
    
    # 2. Detail Enhancement: Edge-aware smoothing + local contrast boost
    enhanced = cv2.detailEnhance(
        denoised,
        sigma_s=10,    # Spatial extent (larger = more smoothing)
        sigma_r=0.15   # Range extent (smaller = more edge preservation)
    )
    return enhanced
```

**Use Case:** When you need quality improvement **without** upscaling (e.g., prep for OCR, web optimization, noise reduction).

---

### 4. Smart Path Resolution

```python
def _resolve_path(path_str: Union[str, Path]) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw
    
    # Walk call stack to find the first non-internal caller
    for frame_info in inspect.stack()[2:]:
        caller_file = Path(frame_info.filename)
        if caller_file.is_file() and not str(caller_file).startswith("<"):
            return (caller_file.parent / raw).resolve()
    
    return (Path.cwd() / raw).resolve()
```

**Why this matters:**
- `enhance("photo.jpg")` in `/home/user/project/script.py` → resolves to `/home/user/project/photo.jpg`
- Works regardless of `cwd` — critical for CLI tools, cron jobs, web services
- No config files, no environment variables, no context managers

---

## Tech Stack Breakdown

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Runtime** | Python | ≥3.9 | Language |
| **Inference** | ONNX Runtime | ≥1.15 | Model execution (CPU/CUDA/CoreML) |
| **Image I/O** | OpenCV (headless) | ≥4.7 | Decode/encode, classical ops |
| **Array Ops** | NumPy | ≥1.22 | Tensor manipulation |
| **Image Format** | Pillow | ≥9.5 | Format support (via OpenCV) |
| **Build** | Hatchling | Latest | PEP 517/518 compliant build |
| **Typing** | mypy + py.typed | Strict | PEP 561 compliance |
| **Testing** | pytest | ≥7.0 | Unit + integration |
| **Formatting** | Black | Latest | Consistent style |
| **MCP Server** | FastMCP + mcp | ≥1.0 | Model Context Protocol server (stdio) for agent integration |
| **Export** | PyTorch + super-image | Dev only | Model → ONNX conversion |

**Dependency Graph:**
```
clarify-ai
├── numpy (array math)
├── opencv-python-headless (image I/O, classical ops)
├── onnxruntime (inference engine) ──▶ CUDA / CoreML / CPU providers
├── pillow (format support)
└── [dev] pytest, mypy, black, build, twine
```

**Zero Heavy ML Frameworks** — No PyTorch, TensorFlow, JAX at runtime.

---

## Packaging & Distribution Strategy

### Build Configuration (pyproject.toml)
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/clarify"]

[tool.hatch.build.targets.wheel.shared-data]
"src/clarify/models" = "clarify/models"  # Models installed to site-packages/clarify/models/
```

### Distribution Artifacts
| Artifact | Size | Contents |
|----------|------|----------|
| `clarify_ai-0.4.0-py3-none-any.whl` | ~35 MB | Python code + bundled ONNX models |
| `clarify_ai-0.4.0.tar.gz` | ~17 MB | Source only |

### Installation Methods
| Method | Command | Use Case |
|--------|---------|----------|
| **PyPI (prod)** | `pip install clarify-ai` | Production, fastest |
| **PyPI + MCP** | `pip install "clarify-ai[mcp]"` | Agent integration (Claude/Cursor) |
| **Editable (dev)** | `pip install -e .[dev]` | Development, testing |
| **Wheel (air-gapped)** | `pip install clarify_ai-0.4.0-py3-none-any.whl` | Offline, air-gapped |
| **Git dep** | `pip install git+https://github.com/...` | CI/CD, pinned versions |
| **Vendor** | `cp -r src/clarify vendor/` | Zero-dependency embedding |

---

## Performance Characteristics

### Latency Benchmarks (CPU, 256×256 input)

| Mode | Scale | First Call | Cached Call | Output Size |
|------|-------|------------|-------------|-------------|
| AI | 2× | ~270 ms | ~300 ms | 512×512 |
| AI | 3× | ~350 ms | ~380 ms | 768×768 |
| AI | 4× | ~450 ms | ~500 ms | 1024×1024 |
| Classical | — | ~300 ms | ~300 ms | 256×256 |

### GPU Performance (Estimated)
| Hardware | 2× Latency | Speedup vs CPU |
|----------|------------|----------------|
| RTX 3080 | ~15 ms | ~18× |
| RTX 4090 | ~8 ms | ~35× |
| M1/M2/M3 (CoreML) | ~25 ms | ~10× |

**Caching Impact:** First call includes model load + session init. Subsequent calls reuse `InferenceSession` — critical for batch processing.

---

## Reliability & Correctness

### Type Safety
```bash
$ mypy src/clarify --strict
Success: no issues found in 4 source files
```
- Full PEP 561 compliance (`py.typed` marker)
- `Literal[2,3,4]` for scale — compile-time validation
- Custom exceptions inherit from stdlib (`FileNotFoundError`, etc.)

### Error Handling
| Exception | Inherits | Trigger |
|-----------|----------|---------|
| `ImageNotFoundError` | `FileNotFoundError` | Input path doesn't exist |
| `UnsupportedImageError` | `ClarifyError` | `cv2.imdecode` returns None |
| `ModelNotFoundError` | `FileNotFoundError` | Bundled `.onnx` missing |
| `ValueError` | — | Invalid `mode` or `scale` |

### Test Coverage
```bash
$ pytest -v --cov=clarify --cov-report=term-missing
# 12 tests covering:
# - AI mode: 2x, 3x, 4x dimension verification
# - Classical mode: dimension preservation
# - Error cases: missing file, bad format, invalid params
# - Input formats: JPEG, PNG, WebP, BMP
# - Path resolution: absolute, relative, custom output
# - CLI integration
```

---

## Deployment Scenarios

### 1. Python Library (Embedded)
```python
from clarify import enhance

# In your image processing pipeline
enhanced = enhance(user_upload.path, mode="ai", scale=4)
```

### 2. CLI Tool (Batch Processing)
```bash
# Process directory
for img in photos/*.jpg; do
    clarify "$img" --scale 2 --output "enhanced/$(basename "$img")"
done
```

### 3. Web Service (FastAPI/Flask)
```python
@app.post("/enhance")
async def enhance_endpoint(file: UploadFile, scale: int = 2):
    temp = save_upload(file)
    result = enhance(temp, mode="ai", scale=scale)
    return FileResponse(result, media_type="image/png")
```

### 4. Air-Gapped / Offline
```bash
# Build wheel on connected machine
python -m build

# Transfer .whl to offline machine
scp dist/clarify_ai-0.4.0-py3-none-any.whl offline-host:

# Install offline
pip install clarify_ai-0.4.0-py3-none-any.whl
# Works completely offline — models bundled
```

---

## Extensibility Points

| Extension Point | Current | Future |
|-----------------|---------|--------|
| **Models** | SPAN, SRVGG Compact, RealPLSKR, EDSR Baseline | Real-ESRGAN full, SwinIR, INT8 quantized bundles |
| **Providers** | CUDA, CoreML, CPU | TensorRT, DirectML, OpenVINO |
| **Modes** | AI, Classical, Denoise, Deblur, Face Restore | Async, concurrent batching |
| **Output** | PNG, JPEG, WebP, TIFF | PDF, HEIC, AVIF |
| **Batching** | Sequential | Async, concurrent.futures |
| **Progress** | None | Callbacks, tqdm integration |
| **Transport** | MCP stdio | Streamable HTTP, WebSocket |

## MCP Server

`clarify-ai` includes a Model Context Protocol (MCP) server (`clarify.mcp_server`) that exposes the local upscaler as tools for Claude Desktop, Cursor, and custom agent runtimes — no image data ever leaves your machine.

### Entry Points

- `clarify` — CLI (`clarify.cli:main`)
- `clarify-mcp` — MCP stdio server (`clarify.mcp_server:main`)

### Tools

| Tool | Description |
|------|-------------|
| `upscale_image` | Enhance/upscale a local image with `arch`, `scale`, `mode`, `format`, `quality`, `strength` |
| `health` | Runtime health: providers, bundled models, cache state, version |
| `list_models` | All model architecture presets with metadata and bundling status |
| `batch_upscale` | Enhance multiple images sequentially with per-item error reporting |

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "clarify": {
      "command": "python",
      "args": ["-m", "clarify.mcp_server"]
    }
  }
}
```

### Example Usage (via MCP client)

```json
{
  "name": "upscale_image",
  "arguments": {
    "image_path": "/path/to/photo.jpg",
    "scale": 2,
    "arch": "span",
    "mode": "ai",
    "output_format": "png",
    "quality": 95
  }
}
```

All inference runs locally via ONNX Runtime + OpenCV in an executor thread, keeping the async event loop responsive.


---

## Competitive Positioning

| Feature | clarify-ai | Real-ESRGAN | basicsr | GFPGAN | OpenCV Only |
|---------|------------|-------------|---------|--------|-------------|
| **Install Size** | 35 MB | 2 GB+ | 2 GB+ | 1 GB+ | 50 MB |
| **PyTorch Required** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Model Bundled** | ✅ | ❌ | ❌ | ❌ | N/A |
| **Offline Ready** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **API Complexity** | 1 function | High | High | Medium | Medium |
| **Hardware Auto** | ✅ | Manual | Manual | Manual | N/A |
| **Path Resolution** | ✅ | Manual | Manual | Manual | Manual |
| **Type Hints** | Strict | Partial | Partial | Partial | ✅ |
| **License** | MIT | BSD-3 | Apache-2 | Apache-2 | BSD |

**Unique Position:** *Only library that combines AI super-resolution + zero-config + offline-ready + typed API in <50 MB.*

---

## Value Proposition

### For Engineering Teams
- **Save 2-4 weeks** of ML infra setup per project
- **Eliminate model drift** — bundled models = reproducible results
- **Reduce container size** from 5 GB → 200 MB (no PyTorch)
- **Zero DevOps** for GPU — works on CPU, scales to GPU automatically

### For Products
- **Ship AI enhancement** in hours, not months
- **Edge deployment ready** — Raspberry Pi, Jetson, mobile (via ONNX)
- **Deterministic costs** — no cloud API fees, no rate limits
- **Privacy compliant** — images never leave user's machine

### For Open Source Portfolio
- Demonstrates: ML ops, ONNX export, Python packaging, CI/CD, type systems
- Real-world usable tool (not a toy demo)
- Clean architecture — extensible, tested, documented

---

## Architecture Summary

```
┌────────────────────────────────────────────────────────────────┐
│  clarify-ai =                                                 │
│                                                                │
│  [EDSR Architecture]                                           │
│       │                                                        │
│       ▼                                                        │
│  [ONNX Export + Dynamic Axes]  ──▶  Universal model format    │
│       │                                                        │
│       ▼                                                        │
│  [ONNX Runtime + Provider Auto-Select]  ──▶  Hardware abstraction │
│       │                                                        │
│       ▼                                                        │
│  [Smart Caching + Path Resolution]  ──▶  Developer experience │
│       │                                                        │
│       ▼                                                        │
│  [Zero-Dependency Wheel + Vendoring]  ──▶  Distribution       │
│                                                                │
│  = Production-ready AI enhancement in one import               │
└────────────────────────────────────────────────────────────────┘
```

**Every layer solves a real bottleneck. No vaporware. No false promises. Just engineering.**

---

## Appendix: Key Files Reference

| File | Purpose |
|------|---------|
| `src/clarify/__init__.py` | Public exports: `enhance`, `health_check`, `MODEL_PRESETS`, `ModelArch`, version |
| `src/clarify/core.py` | Enhancement logic, path resolution, model registry, per-(arch, scale) session caching, inference, health check |
| `src/clarify/cli.py` | CLI entry point (`clarify` command) with `--arch` flag |
| `src/clarify/mcp_server.py` | MCP stdio server: `upscale_image`, `health`, `list_models`, `batch_upscale` |
| `src/clarify/exceptions.py` | Enterprise exception hierarchy with error codes + context |
| `src/clarify/py.typed` | PEP 561 marker |
| `src/clarify/models/` | Bundled ONNX models: `span_x2.onnx`, `realesr-general-x4v3.onnx`, `realplskr_x4.onnx`, `edsr_x2.onnx`, `gfpgan_v1_4.onnx` (optional) |
| `tools/export_span_onnx.py` | PyTorch → ONNX export for SPAN (build-time only) |
| `tools/export_srvgg_onnx.py` | PyTorch → ONNX export for SRVGGNet Compact (build-time only) |
| `tools/export_realplskr_onnx.py` | PyTorch → ONNX export for RealPLSKR (build-time only) |
| `tools/export_gfpgan_onnx.py` | PyTorch → ONNX export for GFPGAN (build-time only) |
| `tools/benchmark.py` | Performance benchmarking |
| `tests/test_clarify.py` | Test suite |
| `.github/workflows/ci.yml` | CI: test matrix + mypy type checking + ruff linting |
| `pyproject.toml` | Build config (hatchling), dependencies, metadata, version 0.4.0 |

---

*Document version: 2.0 | Project: clarify-ai | License: MIT | Model registry: SPAN + SRVGG Compact + RealPLSKR + EDSR Baseline | MCP server included*