"""Tests for the reference CFMNN (blueprint 11.2: cfmnn_core.py).

Key tests: matches the paper's algorithm; 0.5 floor; deterministic ties;
Sum/Count consistency.
"""

import unittest

import numpy as np

from pace_cfmnn.cfmnn_core import CFMNN, feature_terms, ramp
from tests import naive_reference as ref


def random_problem(seed, n=300, d=4, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    # Labels with some structure, so boxes expand as well as get created.
    y = (X[:, 0] * classes + rng.normal(0, 0.3, n)).clip(0, classes - 1).astype(int)
    return X, y


class TestMembership(unittest.TestCase):
    def test_ramp_values(self):
        g = 4.0
        self.assertEqual(ramp(-0.3, g), 1.0)   # inside on this side
        self.assertEqual(ramp(0.0, g), 1.0)
        self.assertAlmostEqual(float(ramp(0.1, g)), 0.6)
        self.assertEqual(ramp(0.25, g), 0.0)    # 1/gamma outside
        self.assertEqual(ramp(0.9, g), 0.0)

    def test_hand_computed_membership(self):
        # Box [0.2, 0.4] x [0.2, 0.4]; point 0.1 outside on feature 0 only.
        v, w = np.array([0.2, 0.2]), np.array([0.4, 0.4])
        x = np.array([0.5, 0.3])
        m = feature_terms(x, v, w, 4.0)
        np.testing.assert_allclose(m, [0.8, 1.0])          # (0.6 + 1)/2 and 1
        self.assertAlmostEqual(float(m.mean()), 0.9)
        self.assertAlmostEqual(ref.membership(list(x), list(v), list(w)), 0.9)

    def test_point_inside_has_membership_one(self):
        v, w = np.array([0.2, 0.1, 0.0]), np.array([0.4, 0.9, 1.0])
        x = np.array([0.3, 0.1, 1.0])  # on the faces counts as inside
        self.assertEqual(feature_terms(x, v, w, 4.0).mean(), 1.0)

    def test_half_floor(self):
        # Every term, and so every membership, lies in [0.5, 1] (blueprint 4.3).
        rng = np.random.default_rng(0)
        lo = rng.uniform(size=(500, 6))
        hi = np.minimum(1.0, lo + rng.uniform(0, 0.5, size=(500, 6)))
        x = rng.uniform(-0.5, 1.5, size=(500, 6))
        m = feature_terms(x, lo, hi, 4.0)
        self.assertGreaterEqual(m.min(), 0.5)
        self.assertLessEqual(m.max(), 1.0)
        # A term reaches the floor exactly once x is 1/gamma = 0.25 outside.
        self.assertEqual(feature_terms(np.array([0.65]), np.array([0.2]),
                                       np.array([0.4]), 4.0)[0], 0.5)


class TestTrainingTrace(unittest.TestCase):
    """Hand-traced runs of Algorithm 2 with theta = 0.3, d = 2 (limit 0.6)."""

    X = np.array([[0.1, 0.1],    # new point box 0
                  [0.2, 0.2],    # expands box 0 to [0.1,0.2]^2 (size 0.2 <= 0.6)
                  [0.15, 0.15],  # inside box 0: membership 1
                  [0.9, 0.9],    # size would be 1.6 > 0.6: new box 1
                  [0.5, 0.5]])   # other class: new box 2
    y = np.array([0, 0, 0, 0, 1])

    def test_original_rule_skips_inside_points(self):
        m = CFMNN(centroid_rule="original").fit(self.X, self.y)
        self.assertEqual(m.n_boxes_, 3)
        np.testing.assert_array_equal(m.V_[0], [0.1, 0.1])
        np.testing.assert_array_equal(m.W_[0], [0.2, 0.2])
        self.assertEqual(m.count_[0], 2)                     # (0.15, 0.15) skipped
        np.testing.assert_allclose(m.centroid_[0], [0.15, 0.15])
        np.testing.assert_array_equal(m.box_class_, [0, 0, 1])
        self.assertEqual(m.fit_stats_["n_inside"], 1)

    def test_fixed_rule_counts_inside_points(self):
        m = CFMNN(centroid_rule="fixed").fit(self.X, self.y)
        self.assertEqual(m.count_[0], 3)
        np.testing.assert_allclose(m.sum_[0], [0.45, 0.45])
        np.testing.assert_allclose(m.centroid_[0], [0.15, 0.15])

    def test_centroid_rules_differ_when_inside_points_are_off_centre(self):
        X = np.array([[0.1, 0.1], [0.3, 0.3], [0.29, 0.29], [0.28, 0.28]])
        y = np.zeros(4, int)
        orig = CFMNN(centroid_rule="original").fit(X, y)
        fixed = CFMNN(centroid_rule="fixed").fit(X, y)
        np.testing.assert_allclose(orig.centroid_[0], [0.2, 0.2])
        np.testing.assert_allclose(fixed.centroid_[0], [0.2425, 0.2425])

    def test_expansion_is_tested_on_the_best_box_only(self):
        # Box A = [0, 0.5] x [0, 0.05] (size 0.55). (0.4, 0.25) is too far for A,
        # so it becomes box B. (0.3, 0.12) is closest to A (membership 0.93) but
        # would make A too big (0.62 > 0.6). It would fit B, yet the paper only
        # tests the best box, so it must become a new box.
        X = np.array([[0.0, 0.0], [0.5, 0.05], [0.4, 0.25], [0.3, 0.12]])
        y = np.zeros(4, int)
        m = CFMNN(expansion="best").fit(X, y)
        self.assertEqual(m.n_boxes_, 3)
        np.testing.assert_array_equal(m.W_[0], [0.5, 0.05])
        np.testing.assert_array_equal(m.V_[2], m.W_[2])  # point box at (0.3, 0.12)

        # The "search" option instead expands box B to take the point.
        s = CFMNN(expansion="search").fit(X, y)
        self.assertEqual(s.n_boxes_, 2)
        np.testing.assert_array_equal(s.V_[1], [0.3, 0.12])
        np.testing.assert_array_equal(s.W_[1], [0.4, 0.25])

    def test_rejects_unscaled_input(self):
        with self.assertRaises(ValueError):
            CFMNN().fit(np.array([[0.2, 1.3]]), [0])
        with self.assertRaises(ValueError):
            CFMNN(centroid_rule="bogus").fit(np.array([[0.2, 0.3]]), [0])


class TestMatchesPaperPseudocode(unittest.TestCase):
    """The fast implementation equals a literal transcription of the paper (G0)."""

    def _compare(self, seed, rule, theta, expansion="best"):
        X, y = random_problem(seed)
        fast = CFMNN(theta=theta, centroid_rule=rule, expansion=expansion,
                     tie_tol=0.0).fit(X, y)
        slow = ref.train(X.tolist(), y.tolist(), theta=theta, fix_centroid=(rule == "fixed"),
                         search=(expansion == "search"))
        self.assertEqual(fast.n_boxes_, len(slow))
        np.testing.assert_array_equal(fast.V_, np.array([b["V"] for b in slow]))
        np.testing.assert_array_equal(fast.W_, np.array([b["W"] for b in slow]))
        np.testing.assert_array_equal(fast.count_, [b["Count"] for b in slow])
        np.testing.assert_allclose(fast.centroid_, np.array([b["Cent"] for b in slow]),
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(fast.classes_[fast.box_class_], [b["dec"] for b in slow])

        Xt, _ = random_problem(seed + 100, n=200)
        Xt = np.vstack([Xt, X[:50]])  # include training points, which often tie
        det = fast.predict_details(Xt)
        for r, x in enumerate(Xt.tolist()):
            dec, box = ref.predict_one(x, slow)
            self.assertEqual(det.y_pred[r], dec)
            self.assertEqual(det.winner[r], box)

    def test_original_rule(self):
        for seed in range(3):
            for theta in (0.1, 0.3, 0.6):
                self._compare(seed, "original", theta)

    def test_fixed_rule(self):
        for seed in range(3):
            for theta in (0.1, 0.3, 0.6):
                self._compare(seed, "fixed", theta)

    def test_search_expansion_option(self):
        for seed in range(2):
            for theta in (0.1, 0.3):
                self._compare(seed, "fixed", theta, expansion="search")


class TestModelInvariants(unittest.TestCase):
    def setUp(self):
        self.X, self.y = random_problem(7, n=600, d=5)

    def test_sum_count_match_members_for_fixed_rule(self):
        m = CFMNN(centroid_rule="fixed").fit(self.X, self.y)
        self.assertEqual(sum(len(mm) for mm in m.members_), len(self.X))
        for j, mm in enumerate(m.members_):
            self.assertEqual(m.count_[j], len(mm))
            np.testing.assert_allclose(m.sum_[j], self.X[mm].sum(axis=0), atol=1e-9)
            self.assertTrue((self.y[mm] == m.classes_[m.box_class_[j]]).all())

    def test_members_and_centroids_lie_inside_their_boxes(self):
        for rule in ("fixed", "original"):
            m = CFMNN(centroid_rule=rule).fit(self.X, self.y)
            for j, mm in enumerate(m.members_):
                self.assertTrue((self.X[mm] >= m.V_[j]).all() and (self.X[mm] <= m.W_[j]).all())
            self.assertTrue((m.centroid_ >= m.V_ - 1e-12).all())
            self.assertTrue((m.centroid_ <= m.W_ + 1e-12).all())

    def test_bounds_and_size_limit(self):
        for expansion in ("best", "search"):
            m = CFMNN(theta=0.3, expansion=expansion).fit(self.X, self.y)
            self.assertTrue((m.V_ <= m.W_).all())
            self.assertTrue((m.V_ >= 0).all() and (m.W_ <= 1).all())
            self.assertTrue(((m.W_ - m.V_).sum(axis=1) <= 5 * 0.3 + 1e-12).all())
            self.assertEqual(sum(len(mm) for mm in m.members_), len(self.X))

    def test_deterministic(self):
        a = CFMNN().fit(self.X, self.y)
        b = CFMNN().fit(self.X, self.y)
        np.testing.assert_array_equal(a.V_, b.V_)
        np.testing.assert_array_equal(a.predict(self.X), b.predict(self.X))

    def test_string_labels(self):
        labels = np.array(["cat", "dog", "emu"])[self.y]
        m = CFMNN().fit(self.X, labels)
        self.assertTrue(set(m.predict(self.X)) <= set(labels))


class TestPredictionDetails(unittest.TestCase):
    def test_tie_goes_to_nearest_centroid(self):
        # Two boxes of different classes both contain x = (0.5, 0.5).
        X = np.array([[0.3, 0.3], [0.6, 0.6],     # class 0 box [0.3,0.6]^2, centroid 0.45
                      [0.4, 0.4], [0.9, 0.9]])    # class 1 box [0.4,0.9]^2, centroid 0.65
        y = np.array([0, 0, 1, 1])
        m = CFMNN(theta=0.5).fit(X, y)
        self.assertEqual(m.n_boxes_, 2)
        det = m.predict_details([[0.5, 0.5], [0.58, 0.58]])
        np.testing.assert_array_equal(det.n_tied, [2, 2])
        np.testing.assert_array_equal(det.y_pred, [0, 1])  # 0.5 nearer 0.45; 0.58 nearer 0.65
        np.testing.assert_array_equal(det.runner_up, [1, 0])

    def test_supports_and_runner_up(self):
        X, y = random_problem(3, n=500, d=4)
        m = CFMNN().fit(X, y)
        det = m.predict_details(X)
        rows = np.arange(len(X))
        self.assertTrue(np.all((det.support_rescaled >= 0) & (det.support_rescaled <= 1)))
        np.testing.assert_allclose(det.membership, m.memberships(X).max(axis=1), atol=1e-9)
        self.assertTrue(np.all(det.best_box[rows, det.cls] == det.winner))
        self.assertTrue(np.all(det.runner_up != det.cls))
        sA = det.support_rescaled[rows, det.cls]
        sB = det.support_rescaled[rows, det.runner_up]
        self.assertTrue(np.all(sA >= sB - 1e-9))
        self.assertTrue(np.all(m.box_class_[det.best_box[rows, det.runner_up]] == det.runner_up))

    def test_class_without_boxes_has_zero_support(self):
        m = CFMNN().fit(np.array([[0.1, 0.1], [0.9, 0.9]]), [0, 1])
        m.class_boxes_[1] = np.array([], dtype=np.int64)  # simulate a class losing its boxes
        det = m.predict_details([[0.85, 0.85]])
        self.assertEqual(det.support_rescaled[0, 1], 0.0)
        self.assertTrue(np.isnan(det.support[0, 1]))

    def test_chunking_does_not_change_results(self):
        X, y = random_problem(5, n=400, d=6)
        a = CFMNN().fit(X, y)
        b = CFMNN(chunk_elems=1).fit(X, y)   # one row per chunk
        da, db = a.predict_details(X), b.predict_details(X)
        np.testing.assert_array_equal(da.winner, db.winner)
        np.testing.assert_array_equal(da.runner_up, db.runner_up)
        np.testing.assert_array_equal(da.support, db.support)

    def test_box_terms_average_to_membership(self):
        X, y = random_problem(9, n=300, d=5)
        m = CFMNN().fit(X, y)
        det = m.predict_details(X)
        terms = m.box_terms(X, det.winner)
        np.testing.assert_allclose(terms.mean(axis=1), det.membership, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
