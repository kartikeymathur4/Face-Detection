import unittest

import numpy as np

from face_recognizer import Matcher
from models import GalleryEntry
from tests.helpers import unit


def entry(uid, name, *vectors):
    return GalleryEntry(uid, name, np.vstack(vectors))


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.gallery = {
            "a": entry("a", "Alice", unit(1, 0)),
            "b": entry("b", "Bob", unit(0, 1)),
        }
        self.matcher = Matcher(threshold=0.5, margin=0.05)

    def test_empty_gallery_is_unknown(self):
        result = Matcher().match(unit(1, 0), {})
        self.assertEqual(result.status, "unknown")

    def test_match_above_threshold(self):
        result = self.matcher.match(unit(1, 0.1), self.gallery)
        self.assertEqual((result.status, result.name), ("match", "Alice"))
        self.assertGreater(result.score, 0.9)

    def test_unknown_below_threshold(self):
        result = self.matcher.match(unit(0, 0, 1), self.gallery)
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.name)

    def test_ambiguous_when_two_users_close(self):
        result = self.matcher.match(unit(1, 1), self.gallery)
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.user_id)

    def test_uses_best_sample_of_user(self):
        gallery = {"a": entry("a", "Alice", unit(0, 1), unit(1, 0))}
        result = self.matcher.match(unit(1, 0), gallery)
        self.assertEqual(result.status, "match")
        self.assertAlmostEqual(result.score, 1.0, places=5)

    def test_threshold_is_configurable(self):
        query = unit(1, 1.5)  # cosine with Alice ~ 0.55
        self.assertEqual(Matcher(0.5, 0.0).match(query, {"a": self.gallery["a"]}).status, "match")
        self.assertEqual(Matcher(0.9, 0.0).match(query, {"a": self.gallery["a"]}).status, "unknown")


if __name__ == "__main__":
    unittest.main()
