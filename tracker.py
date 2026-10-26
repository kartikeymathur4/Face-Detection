"""Lightweight IoU tracker + temporal smoothing of recognition results.

Tracking lets us run the (expensive) embedding model only every N frames per
face and keeps labels stable between runs.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from face_detector import Detection
from models import MatchResult, SmoothedLabel
from utils import iou


@dataclass
class Track:
    track_id: int
    detection: Detection
    history: Deque[MatchResult]
    missed: int = 0
    last_recognized_frame: Optional[int] = None

    def add_result(self, result: MatchResult, frame_index: int) -> None:
        self.history.append(result)
        self.last_recognized_frame = frame_index

    def smoothed(self, min_votes: int) -> SmoothedLabel:
        """Majority vote over the recent window; ties favour 'unknown'."""
        if not self.history:
            return SmoothedLabel("pending")
        keys = [r.user_id if r.status == "match" else None for r in self.history]
        counts = Counter(keys)
        best_key, best_count = max(counts.items(), key=lambda kv: (kv[1], kv[0] is None))
        if best_count < min_votes and len(self.history) < min_votes:
            return SmoothedLabel("pending")
        if best_key is None or best_count < min_votes:
            scores = [r.score for r in self.history if r.status != "match"]
            return SmoothedLabel("unknown", score=max(scores) if scores else 0.0)
        wins = [r for r in self.history if r.user_id == best_key]
        score = sum(r.score for r in wins) / len(wins)
        return SmoothedLabel("match", best_key, wins[-1].name, score)


class FaceTracker:
    def __init__(self, max_age: int = 10, iou_threshold: float = 0.3, window: int = 5) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.window = window
        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """Associate detections with tracks (greedy IoU) and return visible tracks."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                overlap = iou(track.detection.box, det.box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, ti, di))
        pairs.sort(reverse=True)

        used_tracks, used_dets = set(), set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = self.tracks[ti]
            track.detection = detections[di]
            track.missed = 0

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missed += 1

        for di, det in enumerate(detections):
            if di not in used_dets:
                self.tracks.append(
                    Track(self._next_id, det, deque(maxlen=self.window))
                )
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return [t for t in self.tracks if t.missed == 0]
