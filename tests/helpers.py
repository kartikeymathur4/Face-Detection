"""Shared test helpers: settings factory, fake detector/recognizer, synthetic data."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import Settings
from face_detector import Detection


def make_settings(tmp: Path, **overrides) -> Settings:
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["MODELS_DIR"] = str(tmp / "models")
    settings = Settings.from_env()
    return replace(settings, **overrides) if overrides else settings


def make_row(x=200, y=100, w=200, h=240, score=0.95, nose_shift=0.0) -> np.ndarray:
    """A YuNet-style detection row with a frontal face layout."""
    landmarks = [
        x + 0.30 * w, y + 0.35 * h,  # right eye (image left)
        x + 0.70 * w, y + 0.35 * h,  # left eye
        x + (0.50 + nose_shift) * w, y + 0.55 * h,  # nose
        x + 0.35 * w, y + 0.78 * h,  # right mouth corner
        x + 0.65 * w, y + 0.78 * h,  # left mouth corner
    ]
    return np.array([x, y, w, h, *landmarks, score], dtype=np.float32)


def make_detection(**kwargs) -> Detection:
    return Detection.from_row(make_row(**kwargs))


def noisy_frame(seed: int = 0, height: int = 480, width: int = 640) -> np.ndarray:
    """Random-noise image: sharp (high Laplacian variance) with mid brightness."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = list(detections or [])

    def detect(self, frame):
        return list(self.detections)


class FakeRecognizer:
    """Returns the queued vectors one by one (or a fixed vector)."""

    def __init__(self, vectors=None, fixed=None):
        self.queue = [np.asarray(v, dtype=np.float32) for v in (vectors or [])]
        self.fixed = None if fixed is None else np.asarray(fixed, dtype=np.float32)

    def embed(self, frame, detection):
        vec = self.fixed if self.fixed is not None else self.queue.pop(0)
        return vec / np.linalg.norm(vec)


def unit(*values: float, dim: int = 8) -> np.ndarray:
    """Unit vector with the given leading components (rest zero)."""
    v = np.zeros(dim, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)
