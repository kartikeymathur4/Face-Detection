"""Image-processing helpers and small utilities.

Running ``python utils.py`` prompts for an admin password and prints the value
to put in ``ADMIN_PASSWORD_HASH``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from errors import CameraError, ImageDecodeError

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x, y, w, h
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Logging / time
# --------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    """Configure logging once. Never log images, embeddings or names."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_iso(dt: datetime) -> str:
    """Fixed-format UTC timestamp (lexicographically sortable)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Geometry / vectors
# --------------------------------------------------------------------------
def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def clamp_box(box: Box, width: int, height: int) -> Box:
    """Clip an (x, y, w, h) box so it lies inside a width x height image."""
    x, y, w, h = box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (returned as float32). Zero vectors stay zero."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# --------------------------------------------------------------------------
# Image decoding / quality
# --------------------------------------------------------------------------
def decode_image(data: bytes, max_side: int = 1280) -> np.ndarray:
    """Decode image bytes to a BGR array; large images are downscaled."""
    if not data:
        raise ImageDecodeError("The image is empty.")
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageDecodeError("The image could not be decoded.") from exc
    if img is None or img.size == 0:
        raise ImageDecodeError(
            "The image is corrupted or in an unsupported format "
            "(use JPG, PNG, BMP or WebP)."
        )
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness measure: variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class QualityReport:
    ok: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def assess_face_quality(
    frame: np.ndarray,
    row: np.ndarray,
    *,
    min_face_size: int,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
    max_pose_deviation: float,
) -> QualityReport:
    """Check size, sharpness, brightness and frontal pose of a detected face.

    ``row`` is a YuNet detection row: x, y, w, h, 5 landmarks (x, y), score.
    """
    issues: List[str] = []
    metrics: Dict[str, float] = {}
    h_img, w_img = frame.shape[:2]
    x, y, w, h = clamp_box(tuple(int(round(v)) for v in row[:4]), w_img, h_img)

    size = float(min(w, h))
    metrics["face_size"] = size
    if size < min_face_size:
        issues.append(f"Face is too small ({int(size)}px); move closer to the camera.")

    if w > 0 and h > 0:
        crop = cv2.resize(frame[y : y + h, x : x + w], (112, 112), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_variance(gray)
        bright = float(gray.mean())
    else:
        sharp, bright = 0.0, 0.0
    metrics["sharpness"] = sharp
    metrics["brightness"] = bright
    if sharp < blur_threshold:
        issues.append("Image looks blurry; hold still and improve focus.")
    if bright < min_brightness:
        issues.append("Face is too dark; improve the lighting.")
    elif bright > max_brightness:
        issues.append("Face is overexposed; reduce direct light.")

    # Rough frontal-pose check: the nose should sit midway between the eyes.
    lm = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
    left_x, right_x = float(min(lm[0, 0], lm[1, 0])), float(max(lm[0, 0], lm[1, 0]))
    eye_span = right_x - left_x
    if eye_span <= 1.0:
        issues.append("Face is not frontal; look straight at the camera.")
    else:
        ratio = (float(lm[2, 0]) - left_x) / eye_span
        metrics["nose_ratio"] = ratio
        if abs(ratio - 0.5) > max_pose_deviation:
            issues.append("Face is turned too far; look straight at the camera.")

    return QualityReport(ok=not issues, issues=issues, metrics=metrics)


# --------------------------------------------------------------------------
# Drawing / FPS
# --------------------------------------------------------------------------
def draw_face(
    frame: np.ndarray,
    box: Box,
    label: str,
    color: Tuple[int, int, int],
    landmarks: Optional[np.ndarray] = None,
) -> None:
    """Draw a bounding box with a filled label (in place). ASCII text only."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), base = cv2.getTextSize(label, FONT, 0.55, 1)
    top = max(y - th - base - 6, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + base + 6), color, -1)
    cv2.putText(frame, label, (x + 4, top + th + 2), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if landmarks is not None:
        for px, py in np.asarray(landmarks).reshape(-1, 2):
            cv2.circle(frame, (int(px), int(py)), 2, (255, 255, 0), -1)


class FPSCounter:
    """Exponentially smoothed frames-per-second estimate."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._last: Optional[float] = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps

    def reset(self) -> None:
        self._last = None
        self.fps = 0.0


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open the webcam or raise a CameraError with actionable advice."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open camera {index}. Check that a webcam is connected, that no "
            "other application is using it, and that camera access is allowed for your "
            "terminal / Python in the operating-system privacy settings."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):  # some cameras need a few reads to warm up
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.1)
    cap.release()
    raise CameraError(
        "The camera opened but returned no frames. It may be in use by another "
        "application or camera permission may have been denied by the operating system."
    )


# --------------------------------------------------------------------------
# Password hashing (admin access)
# --------------------------------------------------------------------------
_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, iterations: int = _PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------
# Secure deletion
# --------------------------------------------------------------------------
def secure_delete(path: Path) -> None:
    """Overwrite a file with random bytes, then remove it (best effort).

    On SSDs and journaling / copy-on-write file systems an overwrite cannot
    guarantee that old blocks are physically erased; full-disk encryption is
    the robust complement to this step.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    try:
        with open(path, "r+b") as handle:
            handle.write(os.urandom(size))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        logger.warning("Could not overwrite a file before deletion; removing it anyway.")
    path.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    import getpass

    first = getpass.getpass("New admin password: ")
    if len(first) < 8:
        sys.exit("Use at least 8 characters.")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    print("\nAdd this line to your .env file:\n")
    print(f"ADMIN_PASSWORD_HASH={hash_password(first)}")
