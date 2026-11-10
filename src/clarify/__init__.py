"""clarify-ai - Ultra-lightweight AI and classical image enhancement."""

from .core import enhance
from .exceptions import (
    ClarifyError,
    ImageNotFoundError,
    ModelNotFoundError,
    UnsupportedImageError,
)

__all__ = [
    "enhance",
    "ClarifyError",
    "ImageNotFoundError",
    "ModelNotFoundError",
    "UnsupportedImageError",
]

__version__ = "0.2.0"