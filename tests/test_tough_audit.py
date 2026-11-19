"""Tough review B6 and B7: the PACE-SHAP certificate, the churn bound and condition S."""

import copy
import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.audit import churn_bound, impact_set, pace_shap_certificate, training_safe_shrink
from pace_cfmnn.explain import PaceShap
from pace_cfmnn.refine import contains, shrink_box
from tests.test_audit import random_edit
from tests.test_refine import problem


class TestPaceShapCertificate(unittest.TestCase):
    def _check(self, d, exact_max_d, n_edits, seed):
        X, y = problem(seed, n=300, d=d, classes=3)
        m = CFMNN().fit(X, y)
        rng = np.random.default_rng(seed)
        bg = X[rng.choice(len(X), 20, replace=False)]
        X_eval = rng.uniform(size=(25, d))
        X_lab, y_lab = X.copy(), y.copy()
        flagged = rows = 0
        for _ in range(n_edits):
            before = copy.deepcopy(m)
            X_lab, y_lab, removed = random_edit(m, X_lab, y_lab, rng)
            phi_b = PaceShap(before, bg, exact_max_d=exact_max_d, n_permutations=4,
                             seed=3).explain(X_eval)
            phi_a = PaceShap(m, bg, exact_max_d=exact_max_d, n_permutations=4,
                             seed=3).explain(X_eval)
            cert = pace_shap_certificate(before, m, X_eval, bg, removed)
            moved = np.abs(phi_a - phi_b).max(axis=1) > 1e-12
            self.assertFalse((moved & ~cert).any(), (d, removed))
            flagged += cert.sum()
            rows += len(cert)
        return flagged, rows

    def test_exact_mode_values_never_change_outside(self):
        flagged, rows = self._check(d=4, exact_max_d=12, n_edits=30, seed=0)
        self.assertLess(flagged, rows)          # the certificate is not "everything"

    def test_sampled_mode_with_the_same_seed(self):
        flagged, rows = self._check(d=5, exact_max_d=2, n_edits=20, seed=1)
        self.assertLess(flagged, rows)


class TestChurnBound(unittest.TestCase):
    def test_zero_hits(self):
        out = churn_bound(np.zeros(100, dtype=bool), alpha=0.05)
        self.assertAlmostEqual(out["upper"], 1 - 0.05 ** (1 / 100), places=12)

    def test_bounds_the_observed_change(self):
        X, y = problem(2, n=300, d=3, classes=2)
        m = CFMNN().fit(X, y)
        rng = np.random.default_rng(2)
        X_un = rng.uniform(size=(2000, 3))
        X_lab, y_lab = X.copy(), y.copy()
        before = copy.deepcopy(m)
        union = np.zeros(len(X_un), dtype=bool)
        for _ in range(5):
            step = copy.deepcopy(m)
            X_lab, y_lab, removed = random_edit(m, X_lab, y_lab, rng)
            union |= impact_set(step, m, X_un, removed)
        changed = before.predict(X_un) != m.predict(X_un)
        self.assertFalse((changed & ~union).any())
        b = churn_bound(union)
        self.assertGreaterEqual(b["upper"], changed.mean())
        self.assertGreaterEqual(b["upper"], b["share"])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            churn_bound([])


class TestTrainingSafeShrink(unittest.TestCase):
    def test_first_shrinks_at_errors_always_cut_a_member(self):
        """A fitted box is the bounding box of its members, so moving a face cuts the member on it:
        condition S never holds for a first shrink. Self-repair's gate stays necessary."""
        X, y = problem(0, n=400, d=3, classes=3)
        m = CFMNN().fit(X, y)
        yi = np.searchsorted(m.classes_, y)
        det = m.predict_details(X)
        n = 0
        for r in np.flatnonzero(det.cls != yi)[:15]:
            k = int(det.winner[r])
            if not contains(m, k, X[r]):
                continue
            for f in range(3):
                after = copy.deepcopy(m)
                info = shrink_box(after, k, X[r], f, X)
                s = training_safe_shrink(m, after, X, yi, k, removed=info["removed"])
                self.assertFalse(s["safe"])
                self.assertGreater(s["n_slab"], 0)
                n += 1
        self.assertGreater(n, 0)

    def test_safe_second_cuts_break_no_correct_training_prediction(self):
        """After a first shrink a face no longer sits on a member, and a second, small cut of the
        same face can meet S. Whenever S holds, no correct training prediction breaks."""
        n_safe = n_unsafe = 0
        for seed, d in ((0, 3), (1, 4), (2, 5)):
            X, y = problem(seed, n=400, d=d, classes=3)
            m = CFMNN().fit(X, y)
            yi = np.searchsorted(m.classes_, y)
            rng = np.random.default_rng(seed)
            for _ in range(60):
                k = int(rng.integers(m.n_boxes_))
                f = int(rng.integers(d))
                if m.W_[k, f] - m.V_[k, f] < 0.02:
                    continue
                first = copy.deepcopy(m)
                x = first.V_[k] + rng.uniform(0.3, 0.7, size=d) * (first.W_[k] - first.V_[k])
                info = shrink_box(first, k, x, f, X)
                if info["removed"]:
                    continue
                right = first.predict_details(X).cls == yi
                after = copy.deepcopy(first)
                lo_moved = info["side"] == "low"
                step = rng.uniform(0.0005, 0.01)
                if lo_moved:
                    after.V_[k, f] = min(after.V_[k, f] + step, after.W_[k, f])
                else:
                    after.W_[k, f] = max(after.W_[k, f] - step, after.V_[k, f])
                s = training_safe_shrink(first, after, X, yi, k)
                if s["safe"]:
                    n_safe += 1
                    right_after = after.predict_details(X).cls == yi
                    self.assertFalse((right & ~right_after).any(), (seed, k, f))
                else:
                    n_unsafe += 1
        self.assertGreater(n_safe, 0)
        self.assertGreater(n_unsafe, 0)

    def test_clause_iii_a_near_tie_lets_a_third_box_in(self):
        """Without clause (iii), S held here and still a correct prediction broke.

        One feature, gamma = 1, tie_tol = 0.1; the row y = 0.5 is of class 1. Box k (class 0)
        scores 0.95, box j (class 1) 0.94 and box l (class 0) 0.845, just outside the tied group
        (0.95 - 0.1 = 0.85). j's centroid is nearer than k's, so j wins and y is right. Shrinking
        k to 0.905 lowers the top to j's 0.94; the group now reaches 0.84 and lets l in, whose
        centroid is nearest, so y becomes class 0. No row of k's class is cut or dropped.
        """
        V = [[0.6], [0.62], [0.81]]
        W = [[0.7], [0.7], [1.0]]
        m = CFMNN.from_boxes(V, W, box_class=[0, 1, 0], centroid=[[0.69], [0.63], [0.5]], gamma=1.0,
                             tie_tol=0.1)
        X, yi = np.array([[0.5]]), np.array([1])
        self.assertEqual(int(m.predict_details(X).cls[0]), 1)
        after = copy.deepcopy(m)
        after.V_[0, 0] = 0.69
        self.assertEqual(int(after.predict_details(X).cls[0]), 0)            # the prediction broke
        s = training_safe_shrink(m, after, X, yi, 0)
        self.assertEqual((s["n_slab"], s["n_won_dropped"], s["same_centroid"], s["only_this_box"]),
                         (0, 0, True, True))                                 # (i) and (ii) hold
        self.assertEqual(s["n_above_winner"], 1)
        self.assertFalse(s["safe"])
        m0 = CFMNN.from_boxes(V, W, box_class=[0, 1, 0], centroid=[[0.69], [0.63], [0.5]], gamma=1.0,
                              tie_tol=0.0)                                   # with no tolerance, k wins
        self.assertEqual(training_safe_shrink(m0, after, X, np.array([0]), 0)["n_above_winner"], 0)


if __name__ == "__main__":
    unittest.main()
