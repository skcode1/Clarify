"""Custom exception types for clarify-ai."""

from __future__ import annotations


class ClarifyError(Exception):
    """Base exception for all clarify runtime errors."""


class ImageNotFoundError(ClarifyError, FileNotFoundError):
    """Raised when the specified input image does not exist on disk."""
    
    def __init__(self, path: str | None = None, message: str = "The specified input image does not exist on disk."):
        if path:
            message = f"The input image at '{path}' does not exist on disk."
        super().__init__(message)
        self.path = path


class UnsupportedImageError(ClarifyError):
    """Raised when the provided file is unreadable or not an image."""
    
    def __init__(self, path: str | None = None, message: str = "The provided file is unreadable or not a supported image format."):
        if path:
            message = f"The file at '{path}' is unreadable or not a supported image format."
        super().__init__(message)
        self.path = path


class ModelNotFoundError(ClarifyError, FileNotFoundError):
    """Raised when bundled ONNX weights cannot be located."""
    
    def __init__(self, path: str | None = None, message: str = "Bundled ONNX weights cannot be located."):
        if path:
            message = f"Required model file not found at: {path}"
        super().__init__(message)
        self.path = path


class ConfigurationError(ClarifyError, ValueError):
    """Raised when unsupported modes, scales, or format options are provided."""
