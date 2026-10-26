import unittest
from collections import deque

from models import MatchResult
from tests.helpers import make_detection
from tracker import FaceTracker, Track


def match(uid="a", name="Alice", score=0.6):
    return MatchResult("match", uid, name, score)


def unknown(score=0.1):
    return MatchResult("unknown", None, None, score)


class TrackerTests(unittest.TestCase):
    def test_track_persists_when_face_moves_slightly(self):
        tracker = FaceTracker(max_age=3)
        first = tracker.update([make_detection(x=100, y=100)])
        second = tracker.update([make_detection(x=108, y=104)])
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_distant_face_gets_new_track(self):
        tracker = FaceTracker(max_age=3)
        a = tracker.update([make_detection(x=0, y=0, w=100, h=100)])
        b = tracker.update([make_detection(x=400, y=300, w=100, h=100)])
        self.assertNotEqual(a[0].track_id, b[0].track_id)

    def test_two_faces_tracked_independently(self):
        tracker = FaceTracker()
        dets = [make_detection(x=0, y=0, w=100, h=100), make_detection(x=300, y=0, w=100, h=100)]
        ids = {t.track_id for t in tracker.update(dets)}
        again = {t.track_id for t in tracker.update(dets)}
        self.assertEqual(ids, again)
        self.assertEqual(len(ids), 2)

    def test_track_removed_after_max_age(self):
        tracker = FaceTracker(max_age=2)
        tracker.update([make_detection()])
        for _ in range(2):
            self.assertEqual(tracker.update([]), [])
            self.assertEqual(len(tracker.tracks), 1)
        tracker.update([])
        self.assertEqual(tracker.tracks, [])


class SmoothingTests(unittest.TestCase):
    def make_track(self, results, window=5):
        track = Track(1, make_detection(), deque(maxlen=window))
        for i, r in enumerate(results):
            track.add_result(r, i)
        return track

    def test_pending_without_enough_votes(self):
        self.assertEqual(self.make_track([]).smoothed(3).kind, "pending")
        self.assertEqual(self.make_track([match()]).smoothed(3).kind, "pending")

    def test_majority_match_wins_despite_one_outlier(self):
        label = self.make_track([match(), match(), unknown(), match()]).smoothed(3)
        self.assertEqual((label.kind, label.name), ("match", "Alice"))

    def test_unknown_majority(self):
        label = self.make_track([unknown(), unknown(), unknown(), match()]).smoothed(3)
        self.assertEqual(label.kind, "unknown")

    def test_weak_match_is_not_shown(self):
        # Only 2 of 5 votes for Alice (< min_votes) -> treated as unknown
        label = self.make_track([match(), unknown(), match(), unknown(), unknown()]).smoothed(4)
        self.assertEqual(label.kind, "unknown")

    def test_ambiguous_counts_as_unknown(self):
        amb = MatchResult("ambiguous", None, None, 0.5)
        self.assertEqual(self.make_track([amb, amb, amb]).smoothed(3).kind, "unknown")


if __name__ == "__main__":
    unittest.main()
