"""Face embeddings (OpenCV SFace) and cosine-similarity matching.

SFace model: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
License: Apache-2.0. It outputs a 128-D embedding per aligned face. SFace is used
instead of InsightFace/ArcFace weights because the popular pretrained ArcFace
weights carry non-commercial licences and need extra dependencies; SFace runs
on plain OpenCV.

IMPORTANT: similarity scores are cosine similarities, NOT calibrated
probabilities. A score of 0.6 does not mean "60 % sure".
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from errors import ModelFileError
from face_detector import Detection
from models import GalleryEntry, MatchResult
from utils import normalize_vector


class FaceRecognizer:
    def __init__(self, model_path: Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelFileError(
                f"Face recognition model not found at '{path}'. "
                "Run `python download_models.py` to download it."
            )
        try:
            self._model = cv2.FaceRecognizerSF.create(str(path), "")
        except (cv2.error, AttributeError) as exc:
            raise ModelFileError(
                f"Could not load the SFace model '{path.name}': {exc}. "
                "Make sure opencv-python >= 4.9 is installed and the file is not corrupted."
            ) from exc

    def embed(self, frame_bgr: np.ndarray, detection: Detection) -> np.ndarray:
        """Align the face using its landmarks and return an L2-normalised embedding."""
        aligned = self._model.alignCrop(frame_bgr, detection.row)
        feature = self._model.feature(aligned)
        return normalize_vector(feature)


class Matcher:
    """Compare a query embedding with the enrolled gallery (cosine similarity)."""

    def __init__(self, threshold: float = 0.363, margin: float = 0.05) -> None:
        self.threshold = threshold
        self.margin = margin

    def match(self, query: np.ndarray, gallery: Mapping[str, GalleryEntry]) -> MatchResult:
        if not gallery:
            return MatchResult("unknown", None, None, 0.0)
        q = normalize_vector(query)
        scored = []
        for entry in gallery.values():
            # Best score over a user's samples (embeddings are already normalised).
            scored.append((float(np.max(entry.embeddings @ q)), entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        if best_score < self.threshold:
            return MatchResult("unknown", None, None, best_score)
        if len(scored) > 1:
            second_score = scored[1][0]
            if second_score >= self.threshold and (best_score - second_score) < self.margin:
                return MatchResult("ambiguous", None, None, best_score)
        return MatchResult("match", best.user_id, best.name, best_score)
