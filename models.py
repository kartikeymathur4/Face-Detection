"""Small shared data classes (no heavy imports, safe to import anywhere)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class UserRecord:
    """Metadata of an enrolled user. Never contains images or embeddings."""

    user_id: str
    display_name: str
    consent_at: str
    created_at: str
    expires_at: Optional[str]
    sample_count: int


@dataclass(frozen=True, eq=False)
class GalleryEntry:
    """Enrolled identity used for matching: L2-normalised embeddings (N, D)."""

    user_id: str
    name: str
    embeddings: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    """Outcome of comparing one face embedding with the gallery.

    status:
        "match"     - best score passed the threshold and is unambiguous
        "unknown"   - best score is below the threshold (or gallery empty)
        "ambiguous" - two different users both passed and are too close to call
    ``score`` is a cosine similarity, NOT a calibrated probability.
    """

    status: str
    user_id: Optional[str]
    name: Optional[str]
    score: float


@dataclass(frozen=True)
class SmoothedLabel:
    """Temporally smoothed label shown for a tracked face.

    kind: "pending" | "match" | "unknown"
    """

    kind: str
    user_id: Optional[str] = None
    name: Optional[str] = None
    score: float = 0.0
