"""Per-frame pipeline: detect -> track -> (periodically) embed + match -> draw."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import cv2
import numpy as np

from config import Settings
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer, Matcher
from models import GalleryEntry
from tracker import FaceTracker
from utils import FPSCounter, draw_face

logger = logging.getLogger(__name__)

GREEN = (0, 150, 0)
RED = (0, 0, 190)
ORANGE = (0, 130, 240)


@dataclass
class FaceInfo:
    track_id: int
    box: tuple
    kind: str  # "match" | "unknown" | "pending" | "small"
    name: Optional[str]
    score: float


@dataclass
class FrameResult:
    frame: np.ndarray
    faces: List[FaceInfo]
    fps: float
    status: str

    @property
    def face_count(self) -> int:
        return len(self.faces)


class RecognitionPipeline:
    def __init__(
        self,
        settings: Settings,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        gallery: Optional[Mapping[str, GalleryEntry]] = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.recognizer = recognizer
        self.gallery: Mapping[str, GalleryEntry] = gallery or {}
        self.matcher = Matcher(settings.recognition_threshold, settings.ambiguity_margin)
        self.tracker = FaceTracker(
            max_age=settings.max_track_age, window=settings.smoothing_window
        )
        self.fps = FPSCounter()
        self.frame_index = 0
        self.recognize_every = settings.recognize_every_n_frames
        self.show_landmarks = False

    def set_gallery(self, gallery: Mapping[str, GalleryEntry]) -> None:
        self.gallery = gallery

    def reset(self) -> None:
        self.tracker.reset()
        self.fps.reset()
        self.frame_index = 0

    def update_runtime(self, threshold: float, recognize_every: int, show_landmarks: bool) -> None:
        self.matcher.threshold = threshold
        self.recognize_every = max(1, int(recognize_every))
        self.show_landmarks = show_landmarks

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        self.frame_index += 1
        fps = self.fps.tick()
        detections = self.detector.detect(frame_bgr)
        tracks = self.tracker.update(detections)
        output = frame_bgr.copy()
        faces: List[FaceInfo] = []
        min_votes = self.settings.smoothing_min_votes

        for track in tracks:
            box = track.detection.box
            due = (
                track.last_recognized_frame is None
                or self.frame_index - track.last_recognized_frame >= self.recognize_every
            )
            too_small = min(box[2], box[3]) < self.settings.min_live_face_size
            if due and not too_small:
                try:
                    embedding = self.recognizer.embed(frame_bgr, track.detection)
                    track.add_result(self.matcher.match(embedding, self.gallery), self.frame_index)
                except cv2.error:
                    logger.warning("Embedding failed for a face; skipping this frame.")

            label = track.smoothed(min_votes)
            if label.kind == "pending":
                kind = "small" if too_small else "pending"
            else:
                kind = label.kind
            info = FaceInfo(track.track_id, box, kind, label.name, label.score)
            faces.append(info)

            if kind == "match":
                text, color = f"{label.name} ({label.score:.2f})", GREEN
            elif kind == "unknown":
                text, color = "Unknown Person", RED
            elif kind == "small":
                text, color = "Move closer", ORANGE
            else:
                text, color = "Analyzing...", ORANGE
            draw_face(
                output, box, text, color,
                track.detection.landmarks if self.show_landmarks else None,
            )

        return FrameResult(output, faces, fps, self._status(faces))

    def _status(self, faces: List[FaceInfo]) -> str:
        if not faces:
            return "No face detected"
        if not self.gallery:
            return "No users enrolled - every face is shown as Unknown Person"
        counts: Dict[str, int] = {}
        for f in faces:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        names = [f.name for f in faces if f.kind == "match" and f.name]
        parts = []
        if names:
            parts.append("Recognized: " + ", ".join(names))
        if counts.get("unknown"):
            parts.append(f"Unknown: {counts['unknown']}")
        pending = counts.get("pending", 0) + counts.get("small", 0)
        if pending:
            parts.append(f"Analyzing: {pending}")
        return " | ".join(parts)
