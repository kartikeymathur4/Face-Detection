import unittest

import cv2
import numpy as np

from errors import ImageDecodeError
from tests.helpers import make_row, noisy_frame
from utils import (
    FPSCounter,
    assess_face_quality,
    clamp_box,
    decode_image,
    hash_password,
    iou,
    laplacian_variance,
    normalize_vector,
    verify_password,
)

QUALITY_ARGS = dict(
    min_face_size=80, blur_threshold=40.0, min_brightness=40.0,
    max_brightness=220.0, max_pose_deviation=0.2,
)


class GeometryTests(unittest.TestCase):
    def test_iou_identical_and_disjoint(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (50, 50, 10, 10)), 0.0)

    def test_iou_partial_overlap(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)

    def test_clamp_box(self):
        self.assertEqual(clamp_box((-5, -5, 20, 20), 100, 100), (0, 0, 15, 15))
        self.assertEqual(clamp_box((90, 90, 50, 50), 100, 100), (90, 90, 10, 10))

    def test_normalize_vector(self):
        self.assertAlmostEqual(float(np.linalg.norm(normalize_vector([3, 4]))), 1.0, places=5)
        self.assertEqual(float(np.linalg.norm(normalize_vector([0, 0]))), 0.0)


class DecodeTests(unittest.TestCase):
    def test_empty_bytes(self):
        with self.assertRaises(ImageDecodeError):
            decode_image(b"")

    def test_corrupted_bytes(self):
        with self.assertRaises(ImageDecodeError):
            decode_image(b"this is definitely not an image")

    def test_valid_png_roundtrip_and_downscale(self):
        ok, buf = cv2.imencode(".png", noisy_frame(1, 200, 400))
        self.assertTrue(ok)
        img = decode_image(buf.tobytes(), max_side=100)
        self.assertEqual(max(img.shape[:2]), 100)


class QualityTests(unittest.TestCase):
    def test_sharp_frontal_face_passes(self):
        report = assess_face_quality(noisy_frame(2), make_row(), **QUALITY_ARGS)
        self.assertTrue(report.ok, report.issues)

    def test_blurry_face_rejected(self):
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        report = assess_face_quality(flat, make_row(), **QUALITY_ARGS)
        self.assertFalse(report.ok)
        self.assertTrue(any("blurry" in i for i in report.issues))

    def test_dark_face_rejected(self):
        dark = (noisy_frame(3) // 8).astype(np.uint8)
        report = assess_face_quality(dark, make_row(), **{**QUALITY_ARGS, "blur_threshold": 0.0})
        self.assertTrue(any("dark" in i for i in report.issues))

    def test_small_face_rejected(self):
        report = assess_face_quality(noisy_frame(4), make_row(w=40, h=40), **QUALITY_ARGS)
        self.assertTrue(any("small" in i for i in report.issues))

    def test_turned_face_rejected(self):
        report = assess_face_quality(noisy_frame(5), make_row(nose_shift=0.3), **QUALITY_ARGS)
        self.assertTrue(any("turned" in i for i in report.issues))

    def test_laplacian_noise_sharper_than_flat(self):
        noise = cv2.cvtColor(noisy_frame(6), cv2.COLOR_BGR2GRAY)
        flat = np.full_like(noise, 100)
        self.assertGreater(laplacian_variance(noise), laplacian_variance(flat))


class MiscTests(unittest.TestCase):
    def test_password_hash_roundtrip(self):
        stored = hash_password("correct horse", iterations=1000)
        self.assertTrue(verify_password("correct horse", stored))
        self.assertFalse(verify_password("wrong", stored))
        self.assertFalse(verify_password("anything", "not-a-valid-hash"))

    def test_fps_counter_positive(self):
        counter = FPSCounter()
        counter.tick()
        import time
        time.sleep(0.01)
        self.assertGreater(counter.tick(), 0.0)


if __name__ == "__main__":
    unittest.main()
