"""Custom exception types for clarify-ai."""


class ClarifyError(Exception):
    """Base exception for all clarify runtime errors."""


class ImageNotFoundError(ClarifyError, FileNotFoundError):
    """Raised when the specified input image does not exist on disk."""
    
    def __init__(self, path: str = None, message: str = "The specified input image does not exist on disk."):
        if path:
            message = f"The input image at '{path}' does not exist on disk."
        super().__init__(message)
        self.path = path


class UnsupportedImageError(ClarifyError):
    """Raised when the provided file is unreadable or not an image."""
    
    def __init__(self, path: str = None, message: str = "The provided file is unreadable or not a supported image format."):
        if path:
            message = f"The file at '{path}' is unreadable or not a supported image format."
        super().__init__(message)
        self.path = path
