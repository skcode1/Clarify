"""clarify-ai - Ultra-lightweight AI and classical image enhancement."""

from .core import (
    DEFAULT_ARCH,
    DEFAULT_SCALE,
    MODEL_PRESETS,
    ModelArch,
    clear_session_cache,
    enhance,
    health_check,
)
from .exceptions import (
    ClarifyError,
    ConfigurationError,
    ImageNotFoundError,
    ModelNotFoundError,
    UnsupportedImageError,
)

__all__ = [
    "enhance",
    "health_check",
    "MODEL_PRESETS",
    "ModelArch",
    "DEFAULT_ARCH",
    "DEFAULT_SCALE",
    "clear_session_cache",
    "ClarifyError",
    "ConfigurationError",
    "ImageNotFoundError",
    "ModelNotFoundError",
    "UnsupportedImageError",
]

__version__ = "0.4.0"
