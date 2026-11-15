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
    "DEFAULT_ARCH",
    "DEFAULT_SCALE",
    "MODEL_PRESETS",
    "ClarifyError",
    "ConfigurationError",
    "ImageNotFoundError",
    "ModelArch",
    "ModelNotFoundError",
    "UnsupportedImageError",
    "clear_session_cache",
    "enhance",
    "health_check",
]

__version__ = "0.4.0"
