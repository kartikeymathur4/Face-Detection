"""Application-specific exceptions.

Every error that the UI can show to a user derives from ``AppError`` so the
Streamlit layer can catch one type and display ``str(exc)`` safely.
"""


class AppError(Exception):
    """Base class for all expected, user-presentable errors."""


class ConfigError(AppError):
    """A configuration value is missing or invalid."""


class ModelFileError(AppError):
    """A pretrained model file is missing or cannot be loaded."""


class CameraError(AppError):
    """The webcam cannot be opened or does not deliver frames."""


class ImageDecodeError(AppError):
    """An uploaded or captured image is empty, corrupted or unsupported."""


class DatabaseError(AppError):
    """The SQLite database or embedding store failed."""


class DuplicateUserError(AppError):
    """A user with the same display name already exists."""


class RegistrationError(AppError):
    """Enrollment input or samples failed validation."""
