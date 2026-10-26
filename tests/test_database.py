import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

from database import Database
from errors import DatabaseError, DuplicateUserError
from tests.helpers import unit
from utils import utc_now


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = Database(self.tmp / "faces.db", self.tmp / "embeddings")

    def tearDown(self):
        self._tmp.cleanup()

    def embeddings(self):
        return np.vstack([unit(1, 0), unit(1, 0.1)])

    def test_add_list_get(self):
        user = self.db.add_user("Alice", self.embeddings(), retention_days=30)
        self.assertEqual(self.db.count_users(), 1)
        self.assertEqual(self.db.list_users()[0].display_name, "Alice")
        self.assertEqual(self.db.get_user(user.user_id).sample_count, 2)
        self.assertIsNotNone(user.expires_at)

    def test_no_expiry_when_retention_zero(self):
        user = self.db.add_user("Alice", self.embeddings(), retention_days=0)
        self.assertIsNone(user.expires_at)

    def test_duplicate_name_case_insensitive(self):
        self.db.add_user("Alice", self.embeddings())
        with self.assertRaises(DuplicateUserError):
            self.db.add_user("  alice ", self.embeddings())
        self.assertEqual(self.db.count_users(), 1)
        # the failed insert must not leave an orphan embedding file
        self.assertEqual(len(list((self.tmp / "embeddings").glob("*.npy"))), 1)

    def test_load_gallery_is_normalised(self):
        self.db.add_user("Alice", np.array([[3.0, 4.0], [0.0, 2.0]]))
        entry = next(iter(self.db.load_gallery().values()))
        self.assertEqual(entry.name, "Alice")
        np.testing.assert_allclose(np.linalg.norm(entry.embeddings, axis=1), 1.0, rtol=1e-5)

    def test_delete_removes_row_and_file(self):
        user = self.db.add_user("Alice", self.embeddings())
        files = list((self.tmp / "embeddings").glob("*.npy"))
        self.assertEqual(len(files), 1)
        self.assertTrue(self.db.delete_user(user.user_id))
        self.assertEqual(self.db.count_users(), 0)
        self.assertFalse(files[0].exists())
        self.assertFalse(self.db.delete_user(user.user_id))

    def test_purge_expired(self):
        self.db.add_user("Old", self.embeddings(), retention_days=1)
        self.db.add_user("Forever", self.embeddings(), retention_days=0)
        self.assertEqual(self.db.purge_expired(), 0)
        removed = self.db.purge_expired(now=utc_now() + timedelta(days=2))
        self.assertEqual(removed, 1)
        self.assertEqual([u.display_name for u in self.db.list_users()], ["Forever"])

    def test_missing_embedding_file_is_skipped(self):
        user = self.db.add_user("Alice", self.embeddings())
        (self.tmp / "embeddings" / f"{user.user_id}.npy").unlink()
        self.assertEqual(self.db.load_gallery(), {})

    def test_invalid_embeddings_rejected(self):
        with self.assertRaises(ValueError):
            self.db.add_user("Alice", np.zeros((0, 4)))

    def test_init_failure_raises_database_error(self):
        blocker = self.tmp / "file_not_dir"
        blocker.write_text("x")
        with self.assertRaises(DatabaseError):
            Database(blocker / "faces.db", blocker / "embeddings")


if __name__ == "__main__":
    unittest.main()
