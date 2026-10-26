"""Consented enrollment: name validation, sample checks, duplicate prevention.

Raw photos are never stored. A photo is turned into an embedding immediately
and only the embedding vector is kept (in memory first, then in the database).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from config import RESERVED_NAMES, Settings
from database import Database
from errors import DuplicateUserError, RegistrationError
from face_recognizer import Matcher
from models import UserRecord
from utils import assess_face_quality, normalize_rows

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .'_-]{0,49}$")


@dataclass
class SampleReport:
    ok: bool
    message: str
    embedding: Optional[np.ndarray] = None
    issues: List[str] = field(default_factory=list)


class Registrar:
    def __init__(self, settings: Settings, detector, recognizer, database: Database) -> None:
        self.settings = settings
        self.detector = detector
        self.recognizer = recognizer
        self.db = database

    # ------------------------------------------------------------------
    def validate_name(self, name: str) -> str:
        cleaned = " ".join((name or "").split())
        if not cleaned:
            raise RegistrationError("Please enter a display name.")
        if cleaned.casefold() in RESERVED_NAMES:
            raise RegistrationError("That display name is reserved. Choose another one.")
        if not NAME_PATTERN.match(cleaned):
            raise RegistrationError(
                "The display name must be 1-50 characters: English letters, digits, "
                "spaces and . ' _ - only (the on-screen label font supports ASCII)."
            )
        return cleaned

    def evaluate_sample(self, frame_bgr: np.ndarray) -> SampleReport:
        """Validate one captured image and return its embedding if acceptable."""
        detections = self.detector.detect(frame_bgr)
        if not detections:
            return SampleReport(False, "No face detected. Face the camera in good light.")
        if len(detections) > 1:
            return SampleReport(
                False,
                f"{len(detections)} faces detected. Only one person may be in the frame "
                "during enrollment.",
            )
        detection = detections[0]
        report = assess_face_quality(
            frame_bgr,
            detection.row,
            min_face_size=self.settings.min_face_size,
            blur_threshold=self.settings.blur_threshold,
            min_brightness=self.settings.min_brightness,
            max_brightness=self.settings.max_brightness,
            max_pose_deviation=self.settings.max_pose_deviation,
        )
        if not report.ok:
            return SampleReport(False, " ".join(report.issues), issues=report.issues)
        embedding = self.recognizer.embed(frame_bgr, detection)
        return SampleReport(True, "Sample accepted.", embedding=embedding)

    def check_consistency(self, embeddings: np.ndarray) -> None:
        """Every sample must resemble the others (guards against mixed identities)."""
        n = embeddings.shape[0]
        if n < 2:
            return
        for i in range(n):
            others = np.delete(embeddings, i, axis=0).mean(axis=0)
            norm = np.linalg.norm(others)
            if norm == 0:
                raise RegistrationError("Samples are not usable. Please capture them again.")
            similarity = float(embeddings[i] @ (others / norm))
            if similarity < self.settings.sample_consistency_threshold:
                raise RegistrationError(
                    "The captured samples do not look like the same person. "
                    "Clear the samples and capture them again."
                )

    def is_face_already_enrolled(self, embeddings: np.ndarray) -> bool:
        gallery = self.db.load_gallery()
        if not gallery:
            return False
        matcher = Matcher(self.settings.recognition_threshold, margin=0.0)
        return any(matcher.match(e, gallery).status != "unknown" for e in embeddings)

    # ------------------------------------------------------------------
    def enroll(
        self, display_name: str, consent: bool, embeddings: Sequence[np.ndarray]
    ) -> UserRecord:
        name = self.validate_name(display_name)
        if not consent:
            raise RegistrationError("Explicit consent is required before enrollment.")
        count = len(embeddings)
        if count < self.settings.min_enroll_samples:
            raise RegistrationError(
                f"At least {self.settings.min_enroll_samples} valid samples are required "
                f"(you have {count})."
            )
        if count > self.settings.max_enroll_samples:
            raise RegistrationError(
                f"At most {self.settings.max_enroll_samples} samples are allowed."
            )
        matrix = normalize_rows(np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings]))
        self.check_consistency(matrix)

        if self.db.name_exists(name):
            raise DuplicateUserError(
                "A user with this display name already exists. Choose a different name."
            )
        if self.is_face_already_enrolled(matrix):
            raise RegistrationError(
                "This face appears to be enrolled already. Delete the existing enrollment "
                "first if you want to register again."
            )
        return self.db.add_user(name, matrix, retention_days=self.settings.retention_days)
