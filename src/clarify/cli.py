"""Command-line interface for clarify."""

import argparse
import sys
from .core import enhance
from .exceptions import ClarifyError


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clarify",
        description="Instant AI image upscaler and quality enhancement using ONNX Runtime.",
    )
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["ai", "classical"],
        default="ai",
        help="Enhancement pipeline ('ai' or 'classical'). Default: ai",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Scale multiplier for AI mode (2, 3, or 4). Default: 2",
    )
    parser.add_argument("-o", "--output", default=None, help="Custom output path.")

    args = parser.parse_args()

    try:
        dest = enhance(args.image, mode=args.mode, scale=args.scale, output=args.output)
        print(f"Enhanced image saved: {dest}")
    except ClarifyError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as unexpected:
        print(f"Fatal error: {unexpected}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()