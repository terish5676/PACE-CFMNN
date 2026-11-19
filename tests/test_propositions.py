"""Tests for the paper's propositions that had no direct test.

Proposition 1 (tie property): after ``fit``, every training row lies inside a box of its own
class, so every training error is a tie with a wrong-class box that also contains the row.
Proposition 2's lemma (additive coalition value): the membership of a point that takes some
features from x and the rest from z is b(z) plus the sum of x's one-feature changes.

Propositions 2-4 themselves are covered elsewhere: ``test_pace_shap`` (exact values equal brute
force; pruning changes nothing) and ``test_audit`` (no change outside a certificate).
"""

import itertools
import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.cfmnn_core import feature_terms
from pace_cfmnn.refine import shrink_box, shrink_target
from tests.test_refine import problem


def own_and_wrong_box(model, X, y):
    """For each row: is it inside a box of its own class, and inside one of another class?"""
    inside = ((model.V_[None] <= X[:, None, :]) & (X[:, None, :] <= model.W_[None])).all(axis=2)
    own = model.classes_[model.box_class_][None, :] == np.asarray(y)[:, None]
    return (inside & own).any(axis=1), (inside & ~own).any(axis=1)


class TestTieProperty(unittest.TestCase):
    def test_every_training_row_is_inside_an_own_class_box(self):
        """Holds for every theta, both centroid rules, both expansion rules and any row order."""
        for seed, d, classes in ((0, 2, 2), (1, 4, 3), (2, 6, 3)):
            X, y = problem(seed, n=300, d=d, classes=classes)
            for theta, rule, exp in itertools.product((0.05, 0.3, 0.6), ("fixed", "original"),
                                                      ("best", "search")):
                m = CFMNN(theta=theta, centroid_rule=rule, expansion=exp).fit(X, y)
                own, _ = own_and_wrong_box(m, X, y)
                self.assertTrue(own.all(), (seed, theta, rule, exp))
                det = m.predict_details(X)
                self.assertTrue(np.all(det.membership == 1.0))

    def test_every_training_error_is_a_cross_class_tie(self):
        errors = 0
        for seed, d, classes in ((0, 2, 2), (1, 4, 3), (2, 6, 3)):
            X, y = problem(seed, n=300, d=d, classes=classes)
            m = CFMNN(theta=0.5).fit(X, y)          # large boxes: many overlaps
            det = m.predict_details(X)
            wrong = det.y_pred != y
            errors += int(wrong.sum())
            own, other = own_and_wrong_box(m, X, y)
            self.assertTrue(own[wrong].all() and other[wrong].all())
            # The winning box is of the wrong class, holds the row, and the top group is tied.
            for r in np.flatnonzero(wrong):
                k = det.winner[r]
                self.assertTrue((m.V_[k] <= X[r]).all() and (X[r] <= m.W_[k]).all())
                self.assertGreaterEqual(det.n_tied[r], 2)
        self.assertGreater(errors, 0, "the test needs at least one training error")

    def test_the_property_can_fail_after_a_shrink(self):
        """A shrink drops the rows it cuts off, so they may lie in no box of their class."""
        failed = False
        for seed in range(6):
            X, y = problem(seed, n=300, d=2, classes=2)
            m = CFMNN(theta=0.5).fit(X, y)
            det = m.predict_details(X)
            y_idx = np.searchsorted(m.classes_, y)
            for r in np.flatnonzero(det.y_pred != y):
                k = shrink_target(m, X[r], int(y_idx[r]))
                if k < 0:
                    continue
                shrink_box(m, k, X[r], 0, X)
                failed |= not own_and_wrong_box(m, X, y)[0].all()
        self.assertTrue(failed, "expected some shrink to leave a training row outside every own box")


class TestAdditiveCoalitionValue(unittest.TestCase):
    def test_membership_of_a_hybrid_point_is_additive(self):
        rng = np.random.default_rng(0)
        X, y = problem(3, n=200, d=5, classes=3)
        m = CFMNN().fit(X, y)
        d = X.shape[1]
        for _ in range(20):
            x, z = rng.uniform(size=d), rng.uniform(size=d)
            Tx = feature_terms(x, m.V_, m.W_, m.gamma)
            Tz = feature_terms(z, m.V_, m.W_, m.gamma)
            for mask in range(1 << d):
                S = np.array([(mask >> i) & 1 for i in range(d)], dtype=bool)
                hybrid = np.where(S, x, z)
                want = m.memberships(hybrid)[0]
                got = Tz.mean(axis=1) + ((Tx - Tz)[:, S]).sum(axis=1) / d
                np.testing.assert_allclose(got, want, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
