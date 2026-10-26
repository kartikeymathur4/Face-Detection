"""SQLite metadata store + local embedding files.

* ``users`` table: display name, consent/creation/expiry timestamps, sample count.
* Embeddings live in ``<embeddings_dir>/<user_id>.npy`` (float32, shape (N, D)).
* ``audit_log`` records events by random user id only (no names, no biometrics).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from errors import DatabaseError, DuplicateUserError
from models import GalleryEntry, UserRecord
from utils import normalize_rows, secure_delete, to_iso, utc_now

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL,
    name_key       TEXT NOT NULL UNIQUE,
    consent_at     TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    expires_at     TEXT,
    sample_count   INTEGER NOT NULL,
    embedding_file TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    event   TEXT NOT NULL,
    user_id TEXT
);
"""


def make_name_key(display_name: str) -> str:
    """Case-insensitive, whitespace-normalised key used for duplicate detection."""
    return " ".join(display_name.split()).casefold()


class Database:
    def __init__(self, db_path: Path, embeddings_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.embeddings_dir = Path(embeddings_dir)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise DatabaseError(
                f"Could not initialise the database at '{self.db_path}': {exc}. "
                "Check that the data folder exists and is writable."
            ) from exc

    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _audit(conn: sqlite3.Connection, event: str, user_id: Optional[str]) -> None:
        conn.execute(
            "INSERT INTO audit_log (ts, event, user_id) VALUES (?, ?, ?)",
            (to_iso(utc_now()), event, user_id),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            user_id=row["user_id"],
            display_name=row["display_name"],
            consent_at=row["consent_at"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            sample_count=row["sample_count"],
        )

    def _embedding_path(self, user_id: str) -> Path:
        return self.embeddings_dir / f"{user_id}.npy"

    def _write_embeddings(self, path: Path, embeddings: np.ndarray) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as handle:
                np.save(handle, embeddings, allow_pickle=False)
            os.replace(tmp, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise DatabaseError(f"Could not write the embedding file: {exc}") from exc

    # ------------------------------------------------------------------
    def add_user(
        self,
        display_name: str,
        embeddings: np.ndarray,
        retention_days: int = 0,
        consent_at: Optional[datetime] = None,
    ) -> UserRecord:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2-D array (samples x dims).")

        user_id = uuid.uuid4().hex
        created = utc_now()
        expires = created + timedelta(days=retention_days) if retention_days > 0 else None
        path = self._embedding_path(user_id)
        self._write_embeddings(path, emb)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (user_id, display_name, name_key, consent_at, "
                    "created_at, expires_at, sample_count, embedding_file) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        " ".join(display_name.split()),
                        make_name_key(display_name),
                        to_iso(consent_at or created),
                        to_iso(created),
                        to_iso(expires) if expires else None,
                        int(emb.shape[0]),
                        path.name,
                    ),
                )
                self._audit(conn, "enroll", user_id)
        except sqlite3.IntegrityError as exc:
            secure_delete(path)
            raise DuplicateUserError(
                "A user with this display name already exists. Choose a different name."
            ) from exc
        except sqlite3.Error as exc:
            secure_delete(path)
            raise DatabaseError(f"Could not save the user: {exc}") from exc
        record = self.get_user(user_id)
        assert record is not None
        return record

    def name_exists(self, display_name: str) -> bool:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM users WHERE name_key = ?", (make_name_key(display_name),)
                ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc
        return row is not None

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc
        return self._to_record(row) if row else None

    def list_users(self) -> List[UserRecord]:
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM users ORDER BY created_at, display_name").fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc
        return [self._to_record(r) for r in rows]

    def count_users(self) -> int:
        try:
            with self._connect() as conn:
                return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc

    def delete_user(self, user_id: str, event: str = "delete") -> bool:
        """Permanently delete a user: embedding file first, then the DB row."""
        record = self.get_user(user_id)
        if record is None:
            return False
        secure_delete(self._embedding_path(user_id))
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                self._audit(conn, event, user_id)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not delete the user record: {exc}") from exc
        return True

    def purge_expired(self, now: Optional[datetime] = None) -> int:
        """Delete every enrollment whose retention period has ended."""
        now_iso = to_iso(now or utc_now())
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT user_id FROM users WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now_iso,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc
        count = 0
        for row in rows:
            if self.delete_user(row["user_id"], event="purge_expired"):
                count += 1
        return count

    def load_gallery(self) -> Dict[str, GalleryEntry]:
        """Load all enrolled embeddings for matching (rows are L2-normalised)."""
        gallery: Dict[str, GalleryEntry] = {}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT user_id, display_name, embedding_file FROM users"
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc
        for row in rows:
            path = self.embeddings_dir / row["embedding_file"]
            try:
                emb = np.load(path, allow_pickle=False)
                if emb.ndim != 2 or emb.shape[0] == 0:
                    raise ValueError("bad embedding shape")
            except (OSError, ValueError):
                logger.warning("Skipping a user whose embedding file is missing or unreadable.")
                continue
            gallery[row["user_id"]] = GalleryEntry(
                user_id=row["user_id"],
                name=row["display_name"],
                embeddings=normalize_rows(emb),
            )
        return gallery
