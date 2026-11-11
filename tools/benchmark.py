#!/usr/bin/env python
"""Benchmark script for clarify-ai.

Measures inference latency for different modes, scales, and hardware backends.
"""

from __future__ import annotations

import time
import statistics
from pathlib import Path

import cv2
import numpy as np

from clarify import enhance


def create_test_image(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    """Create a synthetic test image."""
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def benchmark_enhance(
    image_path: Path,
    mode: str,
    scale: int = 2,
    runs: int = 5,
    warmup: int = 1,
) -> dict:
    """Benchmark enhance function."""
    # Warmup runs
    for _ in range(warmup):
        enhance(image_path, mode=mode, scale=scale)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = enhance(image_path, mode=mode, scale=scale)
        end = time.perf_counter()
        times.append(end - start)
        # Clean up output
        result.unlink(missing_ok=True)

    return {
        "mode": mode,
        "scale": scale,
        "runs": runs,
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "stdev_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def print_results(results: list[dict]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Mode':<12} {'Scale':<6} {'Runs':<5} {'Mean (ms)':<12} {'Median (ms)':<12} {'Stdev (ms)':<12} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['mode']:<12} {r['scale']:<6} {r['runs']:<5} {r['mean_ms']:<12.2f} {r['median_ms']:<12.2f} {r['stdev_ms']:<12.2f} {r['min_ms']:<10.2f} {r['max_ms']:<10.2f}")
    print("=" * 80 + "\n")


def main() -> None:
    import onnxruntime as ort

    print("clarify-ai Benchmark Suite")
    print("=" * 80)

    # Show available providers
    providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {providers}")

    # Determine active provider
    active_providers = [p for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"] if p in providers]
    print(f"Active provider priority: {active_providers}")

    # Create test images
    test_dir = Path("/tmp/clarify_benchmark")
    test_dir.mkdir(exist_ok=True)

    test_image = test_dir / "test_256.jpg"
    create_test_image(test_image, (256, 256))
    print(f"\nCreated test image: {test_image} (256x256)")

    results = []

    # Benchmark AI mode at different scales
    print("\nBenchmarking AI mode...")
    for scale in [2, 3, 4]:
        print(f"  Scale {scale}x...")
        result = benchmark_enhance(test_image, mode="ai", scale=scale, runs=5, warmup=1)
        results.append(result)

    # Benchmark Classical mode
    print("\nBenchmarking Classical mode...")
    result = benchmark_enhance(test_image, mode="classical", runs=5, warmup=1)
    results.append(result)

    # Print results
    print_results(results)

    # Additional info
    print("Notes:")
    print("- AI mode: EDSR super-resolution via ONNX Runtime")
    print("- Classical mode: OpenCV fastNlMeansDenoisingColored + detailEnhance")
    print("- Classical mode preserves original dimensions (no upscaling)")
    print("- First run includes model loading time; subsequent runs use cached model")
    print(f"- Test image: 256x256, outputs: 512x512 (2x), 768x768 (3x), 1024x1024 (4x)")


if __name__ == "__main__":
    main()