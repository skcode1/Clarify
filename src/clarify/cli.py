"""Command-line interface for clarify-ai."""

import argparse
import sys
from pathlib import Path
from clarify.core import enhance
from clarify.exceptions import ClarifyError


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clarify",
        description="Upscale and enhance images with EDSR super-resolution.",
    )
    parser.add_argument("image", help="Target image path to enhance.")
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Upscale factor (2, 3, or 4). Default: 2",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional custom destination filename or path.",
    )

    args = parser.parse_args()

    try:
        result_path = enhance(args.image, scale=args.scale, output=args.output)
        print(f"Enhanced image saved: {result_path}")
    except ClarifyError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as unexpected:
        print(f"Fatal error: {unexpected}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()