import tempfile
import unittest
from pathlib import Path

import numpy as np

from database import Database
from errors import DuplicateUserError, RegistrationError
from registration import Registrar
from tests.helpers import (
    FakeDetector,
    FakeRecognizer,
    make_detection,
    make_settings,
    noisy_frame,
    unit,
)


class RegistrarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.settings = make_settings(self.tmp)
        self.db = Database(self.settings.db_path, self.settings.embeddings_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def registrar(self, detections=None, vectors=None):
        return Registrar(
            self.settings,
            FakeDetector(detections if detections is not None else [make_detection()]),
            FakeRecognizer(vectors=vectors or [unit(1, 0)]),
            self.db,
        )

    def good_samples(self, base=(1, 0)):
        return [unit(*base), unit(base[0], base[1] + 0.1), unit(base[0], base[1] - 0.1)]

    # ---- name / consent / counts ---------------------------------------
    def test_empty_name_rejected(self):
        with self.assertRaises(RegistrationError):
            self.registrar().enroll("   ", True, self.good_samples())

    def test_invalid_and_reserved_names_rejected(self):
        for bad in ("Unknown Person", "bad<script>", "x" * 60, "नाम"):
            with self.assertRaises(RegistrationError, msg=bad):
                self.registrar().validate_name(bad)

    def test_consent_required(self):
        with self.assertRaises(RegistrationError):
            self.registrar().enroll("Alice", False, self.good_samples())

    def test_too_few_and_too_many_samples(self):
        with self.assertRaises(RegistrationError):
            self.registrar().enroll("Alice", True, self.good_samples()[:2])
        many = [unit(1, 0)] * (self.settings.max_enroll_samples + 1)
        with self.assertRaises(RegistrationError):
            self.registrar().enroll("Alice", True, many)

    # ---- sample evaluation ------------------------------------------------
    def test_no_face_detected(self):
        report = self.registrar(detections=[]).evaluate_sample(noisy_frame())
        self.assertFalse(report.ok)
        self.assertIn("No face", report.message)

    def test_multiple_faces_rejected(self):
        dets = [make_detection(x=50), make_detection(x=350)]
        report = self.registrar(detections=dets).evaluate_sample(noisy_frame())
        self.assertFalse(report.ok)
        self.assertIn("2 faces", report.message)

    def test_low_quality_rejected(self):
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        report = self.registrar().evaluate_sample(flat)
        self.assertFalse(report.ok)
        self.assertTrue(report.issues)

    def test_good_sample_returns_embedding(self):
        report = self.registrar().evaluate_sample(noisy_frame())
        self.assertTrue(report.ok, report.message)
        self.assertEqual(report.embedding.shape, (8,))

    # ---- enrollment ----------------------------------------------------
    def test_successful_enrollment(self):
        user = self.registrar().enroll("Alice", True, self.good_samples())
        self.assertEqual(user.display_name, "Alice")
        self.assertEqual(self.db.count_users(), 1)
        self.assertEqual(user.sample_count, 3)

    def test_inconsistent_samples_rejected(self):
        mixed = [unit(1, 0), unit(0, 1), unit(0, 0, 1)]
        with self.assertRaises(RegistrationError):
            self.registrar().enroll("Alice", True, mixed)
        self.assertEqual(self.db.count_users(), 0)

    def test_duplicate_name_rejected(self):
        self.registrar().enroll("Alice", True, self.good_samples())
        with self.assertRaises(DuplicateUserError):
            self.registrar().enroll("alice", True, self.good_samples((0, 1)))

    def test_duplicate_face_rejected(self):
        self.registrar().enroll("Alice", True, self.good_samples())
        with self.assertRaises(RegistrationError) as ctx:
            self.registrar().enroll("Alicia", True, self.good_samples())
        self.assertIn("already", str(ctx.exception))
        self.assertEqual(self.db.count_users(), 1)

    def test_different_person_can_enroll(self):
        self.registrar().enroll("Alice", True, self.good_samples((1, 0)))
        user = self.registrar().enroll("Bob", True, self.good_samples((0, 1)))
        self.assertEqual(user.display_name, "Bob")
        self.assertEqual(self.db.count_users(), 2)


if __name__ == "__main__":
    unittest.main()
