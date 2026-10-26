import tempfile
import unittest
from pathlib import Path

from models import GalleryEntry
from pipeline import RecognitionPipeline
from tests.helpers import (
    FakeDetector,
    FakeRecognizer,
    make_detection,
    make_settings,
    noisy_frame,
    unit,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(
            Path(self._tmp.name), recognize_every_n_frames=1, smoothing_window=3, smoothing_min_votes=2
        )

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, vector, gallery=None):
        return RecognitionPipeline(
            self.settings,
            FakeDetector([make_detection()]),
            FakeRecognizer(fixed=vector),
            gallery or {},
        )

    def run_frames(self, pipeline, n=3):
        frame = noisy_frame()
        result = None
        for _ in range(n):
            result = pipeline.process(frame)
        return result

    def test_no_face(self):
        pipeline = self.build(unit(1, 0))
        pipeline.detector.detections = []
        result = pipeline.process(noisy_frame())
        self.assertEqual(result.face_count, 0)
        self.assertEqual(result.status, "No face detected")

    def test_enrolled_face_is_recognised_after_votes(self):
        gallery = {"a": GalleryEntry("a", "Alice", unit(1, 0).reshape(1, -1))}
        result = self.run_frames(self.build(unit(1, 0.05), gallery))
        self.assertEqual(result.faces[0].kind, "match")
        self.assertEqual(result.faces[0].name, "Alice")
        self.assertIn("Alice", result.status)

    def test_stranger_is_unknown(self):
        gallery = {"a": GalleryEntry("a", "Alice", unit(1, 0).reshape(1, -1))}
        result = self.run_frames(self.build(unit(0, 0, 1), gallery))
        self.assertEqual(result.faces[0].kind, "unknown")

    def test_empty_gallery_everyone_unknown(self):
        result = self.run_frames(self.build(unit(1, 0)))
        self.assertEqual(result.faces[0].kind, "unknown")
        self.assertIn("No users enrolled", result.status)

    def test_output_frame_is_a_copy_with_overlay(self):
        pipeline = self.build(unit(1, 0))
        frame = noisy_frame()
        original = frame.copy()
        result = pipeline.process(frame)
        self.assertTrue((frame == original).all())
        self.assertFalse((result.frame == original).all())


if __name__ == "__main__":
    unittest.main()
