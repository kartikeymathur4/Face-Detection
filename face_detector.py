"""Face detection with OpenCV's YuNet (cv2.FaceDetectorYN).

YuNet model: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
License: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from errors import ModelFileError
from utils import Box


@dataclass(frozen=True, eq=False)
class Detection:
    """One detected face.

    ``row`` is a float32 array of 15 values in ORIGINAL frame coordinates:
    x, y, w, h, right-eye(x,y), left-eye(x,y), nose(x,y), right-mouth(x,y),
    left-mouth(x,y), score. The recogniser needs this exact layout.
    """

    row: np.ndarray

    @classmethod
    def from_row(cls, row: np.ndarray) -> "Detection":
        arr = np.ascontiguousarray(np.asarray(row, dtype=np.float32).reshape(-1))
        if arr.shape[0] != 15:
            raise ValueError("A detection row must contain 15 values.")
        return cls(row=arr)

    @property
    def box(self) -> Box:
        x, y, w, h = self.row[:4]
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    @property
    def score(self) -> float:
        return float(self.row[14])

    @property
    def landmarks(self) -> np.ndarray:
        return self.row[4:14].reshape(5, 2)


class FaceDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        input_width: int = 640,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelFileError(
                f"Face detector model not found at '{path}'. "
                "Run `python download_models.py` to download it."
            )
        self.input_width = input_width
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold, nms_threshold, top_k
            )
        except (cv2.error, AttributeError) as exc:
            raise ModelFileError(
                f"Could not load the YuNet model '{path.name}': {exc}. "
                "Make sure opencv-python >= 4.9 is installed and the file is not corrupted "
                "(delete it and run `python download_models.py` again)."
            ) from exc

    def set_score_threshold(self, value: float) -> None:
        self._detector.setScoreThreshold(float(value))

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect faces; results are in the coordinates of ``frame_bgr``."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.input_width / float(w))
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        sh, sw = small.shape[:2]
        self._detector.setInputSize((sw, sh))
        _, faces = self._detector.detect(small)
        if faces is None:
            return []
        detections: List[Detection] = []
        for face in faces:
            row = np.array(face, dtype=np.float32).reshape(-1)[:15]
            row[:14] /= scale  # box + landmarks back to original coordinates
            detections.append(Detection.from_row(row))
        return detections
