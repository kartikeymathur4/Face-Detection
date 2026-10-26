"""Application settings.

Values are read from environment variables (optionally from a ``.env`` file in
the project folder). See ``.env.example`` for every available option.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from errors import ConfigError

try:  # python-dotenv is optional at import time; requirements.txt installs it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent

RESERVED_NAMES = {"unknown", "unknown person"}


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc


def _get_float(name: str, default: float) -> float:
    raw = _get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}.")


def _get_admin_password_hash() -> str:
    value = os.environ.get("ADMIN_PASSWORD_HASH", "").strip()
    if value:
        return value

    try:
        value = str(st.secrets.get("ADMIN_PASSWORD_HASH", "")).strip()
    except StreamlitSecretNotFoundError:
        value = ""
    if value:
        return value

    if load_dotenv is not None:
        from dotenv import dotenv_values

        return str(dotenv_values(BASE_DIR / ".env").get("ADMIN_PASSWORD_HASH") or "").strip()
    return ""


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


@dataclass(frozen=True)
class Settings:
    # --- paths -----------------------------------------------------------
    data_dir: Path
    models_dir: Path
    detector_model: str
    recognizer_model: str
    # --- camera ----------------------------------------------------------
    camera_index: int
    camera_width: int
    camera_height: int
    mirror_preview: bool
    # --- detection -------------------------------------------------------
    detect_input_width: int
    detect_score_threshold: float
    detect_nms_threshold: float
    detect_top_k: int
    # --- recognition -----------------------------------------------------
    recognition_threshold: float
    ambiguity_margin: float
    recognize_every_n_frames: int
    max_track_age: int
    smoothing_window: int
    smoothing_min_votes: int
    min_live_face_size: int
    # --- enrollment quality ---------------------------------------------
    min_face_size: int
    blur_threshold: float
    min_brightness: float
    max_brightness: float
    max_pose_deviation: float
    min_enroll_samples: int
    max_enroll_samples: int
    sample_consistency_threshold: float
    max_image_side: int
    # --- privacy / security ---------------------------------------------
    retention_days: int
    admin_password_hash: str
    require_admin_for_enrollment: bool
    log_level: str

    # ------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "faces.db"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def detector_model_path(self) -> Path:
        return self.models_dir / self.detector_model

    @property
    def recognizer_model_path(self) -> Path:
        return self.models_dir / self.recognizer_model

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        if load_dotenv is not None:
            load_dotenv(BASE_DIR / ".env", override=False)
        settings = cls(
            data_dir=_resolve(_get("DATA_DIR", "data")),
            models_dir=_resolve(_get("MODELS_DIR", "models")),
            detector_model=_get("DETECTOR_MODEL", "face_detection_yunet_2023mar.onnx"),
            recognizer_model=_get("RECOGNIZER_MODEL", "face_recognition_sface_2021dec.onnx"),
            camera_index=_get_int("CAMERA_INDEX", 0),
            camera_width=_get_int("CAMERA_WIDTH", 640),
            camera_height=_get_int("CAMERA_HEIGHT", 480),
            mirror_preview=_get_bool("MIRROR_PREVIEW", True),
            detect_input_width=_get_int("DETECT_INPUT_WIDTH", 640),
            detect_score_threshold=_get_float("DETECT_SCORE_THRESHOLD", 0.8),
            detect_nms_threshold=_get_float("DETECT_NMS_THRESHOLD", 0.3),
            detect_top_k=_get_int("DETECT_TOP_K", 50),
            recognition_threshold=_get_float("RECOGNITION_THRESHOLD", 0.363),
            ambiguity_margin=_get_float("AMBIGUITY_MARGIN", 0.05),
            recognize_every_n_frames=_get_int("RECOGNIZE_EVERY_N_FRAMES", 5),
            max_track_age=_get_int("MAX_TRACK_AGE", 10),
            smoothing_window=_get_int("SMOOTHING_WINDOW", 5),
            smoothing_min_votes=_get_int("SMOOTHING_MIN_VOTES", 3),
            min_live_face_size=_get_int("MIN_LIVE_FACE_SIZE", 40),
            min_face_size=_get_int("MIN_FACE_SIZE", 80),
            blur_threshold=_get_float("BLUR_THRESHOLD", 40.0),
            min_brightness=_get_float("MIN_BRIGHTNESS", 40.0),
            max_brightness=_get_float("MAX_BRIGHTNESS", 220.0),
            max_pose_deviation=_get_float("MAX_POSE_DEVIATION", 0.2),
            min_enroll_samples=_get_int("MIN_ENROLL_SAMPLES", 3),
            max_enroll_samples=_get_int("MAX_ENROLL_SAMPLES", 8),
            sample_consistency_threshold=_get_float("SAMPLE_CONSISTENCY_THRESHOLD", 0.4),
            max_image_side=_get_int("MAX_IMAGE_SIDE", 1280),
            retention_days=_get_int("RETENTION_DAYS", 365),
            admin_password_hash=_get_admin_password_hash(),
            require_admin_for_enrollment=_get_bool("REQUIRE_ADMIN_FOR_ENROLLMENT", True),
            log_level=_get("LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        problems = []
        if not 0.0 < self.recognition_threshold < 1.0:
            problems.append("RECOGNITION_THRESHOLD must be between 0 and 1.")
        if not 0.0 <= self.ambiguity_margin < 0.5:
            problems.append("AMBIGUITY_MARGIN must be between 0 and 0.5.")
        if not 0.0 < self.detect_score_threshold <= 1.0:
            problems.append("DETECT_SCORE_THRESHOLD must be in (0, 1].")
        if self.detect_input_width < 160:
            problems.append("DETECT_INPUT_WIDTH must be at least 160.")
        if self.recognize_every_n_frames < 1:
            problems.append("RECOGNIZE_EVERY_N_FRAMES must be >= 1.")
        if self.smoothing_window < 1 or self.smoothing_min_votes < 1:
            problems.append("SMOOTHING_WINDOW and SMOOTHING_MIN_VOTES must be >= 1.")
        if self.smoothing_min_votes > self.smoothing_window:
            problems.append("SMOOTHING_MIN_VOTES cannot exceed SMOOTHING_WINDOW.")
        if self.min_enroll_samples < 1 or self.max_enroll_samples < self.min_enroll_samples:
            problems.append("Enrollment sample limits are inconsistent.")
        if self.retention_days < 0:
            problems.append("RETENTION_DAYS must be >= 0 (0 = keep until deleted).")
        if self.min_brightness >= self.max_brightness:
            problems.append("MIN_BRIGHTNESS must be below MAX_BRIGHTNESS.")
        if problems:
            raise ConfigError("Invalid configuration: " + " ".join(problems))


def privacy_notice_text(settings: Settings) -> str:
    """Human-readable biometric data notice shown in the UI."""
    if settings.retention_days > 0:
        retention = (
            f"Enrollments are automatically deleted {settings.retention_days} days "
            "after they were created."
        )
    else:
        retention = "Enrollments are kept until an administrator deletes them."
    return (
        "**Biometric data notice**\n\n"
        "- **What is collected:** a display name, the time you gave consent, and "
        "numerical face embeddings (128 numbers per sample) computed from the "
        "webcam photos you provide during enrollment.\n"
        "- **What is NOT kept:** the enrollment photos themselves are processed in "
        "memory and discarded. Live camera frames are never saved.\n"
        "- **Where it is stored:** only on this computer (SQLite database + local "
        "embedding files). Nothing is uploaded to external services.\n"
        "- **How it is used:** solely to label faces in the live view of this "
        "application. Embeddings are not used for any other purpose.\n"
        f"- **Retention:** {retention}\n"
        "- **Deletion:** an administrator can delete any enrollment at any time; "
        "the embedding file is overwritten before removal (best effort) and the "
        "database record is removed.\n"
        "- **Consent:** enrollment requires your explicit consent. Do not enroll "
        "or identify anyone without their knowledge and permission."
    )
