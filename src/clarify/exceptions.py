"""Custom exception types for clarify-ai with enterprise error codes and context."""

from __future__ import annotations

from typing import Optional


class ClarifyError(Exception):
    """Base exception for all clarify runtime errors."""

    code: str = "CLARIFY_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.context = context or {}


class ImageNotFoundError(ClarifyError, FileNotFoundError):
    """Raised when the specified input image does not exist on disk."""

    code = "IMAGE_NOT_FOUND"

    def __init__(
        self,
        path: Optional[str] = None,
        message: str = "The specified input image does not exist on disk.",
    ) -> None:
        if path:
            message = f"The input image at '{path}' does not exist on disk."
        super().__init__(message, code=self.code, context={"path": path})
        self.path = path


class UnsupportedImageError(ClarifyError):
    """Raised when the provided file is unreadable or not an image."""

    code = "UNSUPPORTED_IMAGE"

    def __init__(
        self,
        path: Optional[str] = None,
        message: str = "The provided file is unreadable or not a supported image format.",
    ) -> None:
        if path:
            message = f"The file at '{path}' is unreadable or not a supported image format."
        super().__init__(message, code=self.code, context={"path": path})
        self.path = path


class ModelNotFoundError(ClarifyError, FileNotFoundError):
    """Raised when bundled ONNX weights cannot be located."""

    code = "MODEL_NOT_FOUND"

    def __init__(
        self,
        path: Optional[str] = None,
        message: str = "Bundled ONNX weights cannot be located.",
    ) -> None:
        if path:
            message = f"Required model file not found at: {path}"
        super().__init__(message, code=self.code, context={"path": path})
        self.path = path


class ConfigurationError(ClarifyError, ValueError):
    """Raised when unsupported modes, scales, or format options are provided."""

    code = "CONFIGURATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(message, code=self.code, context=context)


class InferenceError(ClarifyError):
    """Raised when ONNX Runtime inference fails."""

    code = "INFERENCE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(message, code=self.code, context=context)


class EncodingError(ClarifyError):
    """Raised when image encoding/saving fails."""

    code = "ENCODING_ERROR"

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(message, code=self.code, context=context)
