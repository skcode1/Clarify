"""clarify - 3-line AI image upscaling and enhancement."""

from clarify.core import enhance
from clarify.exceptions import (
    ClarifyError,
    ImageNotFoundError,
    UnsupportedImageError,
)

__all__ = ["enhance", "ClarifyError", "ImageNotFoundError", "UnsupportedImageError"]
__version__ = "0.1.0"