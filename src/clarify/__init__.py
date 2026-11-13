"""clarify-ai - Ultra-lightweight AI and classical image enhancement."""

from .core import enhance
from .exceptions import (
    ClarifyError,
    ConfigurationError,
    ImageNotFoundError,
    ModelNotFoundError,
    UnsupportedImageError,
)
from .models import (
    Precision,
    get_edsr_model_path,
    list_available_models,
)

__all__ = [
    "ClarifyError",
    "ConfigurationError",
    "ImageNotFoundError",
    "ModelNotFoundError",
    "Precision",
    "UnsupportedImageError",
    "enhance",
    "get_edsr_model_path",
    "list_available_models",
]

__version__ = "0.3.0"
