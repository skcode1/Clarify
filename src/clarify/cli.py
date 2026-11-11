"""Command-line interface for clarify."""

import argparse
import sys
from .core import enhance
from .exceptions import ClarifyError


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clarify",
        description="Unified AI & Classical Image Enhancement Tool via ONNX Runtime.",
    )
    parser.add_argument("image", help="Target input image path.")
    parser.add_argument(
        "-m", "--mode",
        choices=["ai", "classical", "denoise", "deblur", "face_restore"],
        default="ai",
        help="Enhancement pipeline (default: ai).",
    )
    parser.add_argument(
        "-s", "--scale",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Super-resolution scale factor for 'ai' mode (default: 2).",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["png", "jpeg", "webp", "tiff"],
        default="png",
        help="Target output container format (default: png).",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        help="Encoding quality (1-100) for JPEG and WebP (default: 95).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Filtering strength for denoise/deblur (default: 1.0).",
    )
    parser.add_argument("-o", "--output", default=None, help="Explicit target file path.")

    args = parser.parse_args()

    try:
        saved_path = enhance(
            args.image,
            mode=args.mode,
            scale=args.scale,
            output=args.output,
            output_format=args.format,
            quality=args.quality,
            strength=args.strength,
        )
        print(f"Processed image: {saved_path}")
    except ClarifyError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
