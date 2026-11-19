"""Tests for the edit certificate and the churn measures (pace_cfmnn.audit)."""

import copy
import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.audit import (changed_boxes, explanation_churn, impact_set, in_top_group,
                              prediction_churn, top_groups, top_k_sets)
from pace_cfmnn.refine import shrink_box
from tests.test_refine import overlap_case, problem


def random_edit(m, X_lab, y_lab, rng):
    """One random shrink (of a box around a point inside it) or add, in place.

    Returns (X_lab, y_lab, index of a removed box or None).
    """
    d = m.n_features_in_
    if rng.uniform() < 0.6:
        k = int(rng.integers(m.n_boxes_))
        x = m.V_[k] + rng.uniform(size=d) * (m.W_[k] - m.V_[k])
        info = shrink_box(m, k, x, int(rng.integers(d)), X_lab)
        return X_lab, y_lab, (info["box"] if info["removed"] else None)
    x, t = rng.uniform(size=d), rng.choice(m.classes_)
    X_lab, y_lab = np.vstack([X_lab, x]), np.append(y_lab, t)
    m.add_points(x, [t], rows=[len(X_lab) - 1])
    return X_lab, y_lab, None


class TestCertificate(unittest.TestCase):
    def test_every_prediction_change_is_inside_the_impact_set(self):
        """200 random edits on 2-D and 4-D data: no change ever falls outside either certificate."""
        changed_total = flagged_total = tight_total = rows_total = removals = 0
        for seed, d, classes in ((0, 2, 2), (1, 4, 3)):
            X, y = problem(seed, n=300, d=d, classes=classes)
            m = CFMNN().fit(X, y)
            rng = np.random.default_rng(seed)
            X_eval = np.vstack([rng.uniform(size=(400, d)), X])   # training rows give many ties
            X_lab, y_lab = X.copy(), y.copy()
            for _ in range(100):
                before = copy.deepcopy(m)
                X_lab, y_lab, removed = random_edit(m, X_lab, y_lab, rng)
                removals += removed is not None
                changed = before.predict(X_eval) != m.predict(X_eval)
                flagged = impact_set(before, m, X_eval, removed)
                tight = impact_set(before, m, X_eval, removed, tight=True)
                self.assertFalse((changed & ~tight).any(), (seed, d, removed))
                self.assertFalse((tight & ~flagged).any())   # the tight one is inside the other
                changed_total += changed.sum()
                flagged_total += flagged.sum()
                tight_total += tight.sum()
                rows_total += len(X_eval)
        self.assertGreater(removals, 0)                    # removals were exercised
        self.assertGreater(changed_total, 0)
        self.assertLess(flagged_total, 0.5 * rows_total)   # and the certificate is far from "everything"
        self.assertLess(tight_total, flagged_total)

    def test_changed_boxes(self):
        m, X, y, x, _, _ = overlap_case()
        X_lab = np.vstack([X, x])
        a = copy.deepcopy(m)
        shrink_box(a, 0, x, 0, X_lab)
        cb, ca = changed_boxes(m, a)
        self.assertEqual((cb.tolist(), ca.tolist()), ([0], [0]))
        b = copy.deepcopy(m)
        b.add_points(np.array([0.0, 1.0]), [1], rows=[len(X)])     # too far to absorb: a new box
        cb, ca = changed_boxes(m, b)
        self.assertEqual((cb.tolist(), ca.tolist()), ([], [m.n_boxes_]))
        c = copy.deepcopy(m)
        c._remove_box(0)
        cb, ca = changed_boxes(m, c, removed=0)
        self.assertEqual((cb.tolist(), ca.tolist()), ([0], []))
        with self.assertRaises(ValueError):
            changed_boxes(m, c)                              # a removal not declared
        self.assertFalse(impact_set(m, m, X).any())         # no edit, no impact

    def test_top_groups_match_predict_details(self):
        for seed in range(3):
            X, y = problem(seed, n=500)
            m = CFMNN().fit(X[:300], y[:300])
            det = m.predict_details(X)
            tied, cls = top_groups(m, X)
            np.testing.assert_array_equal(cls, det.cls)
            np.testing.assert_array_equal(tied.sum(axis=1), det.n_tied)

    def test_in_top_group_matches_the_tie_rule(self):
        m, X, y, x, _, _ = overlap_case()
        det = m.predict_details(X)
        for j in range(m.n_boxes_):
            hit = in_top_group(m, X, [j])
            self.assertTrue(hit[det.winner == j].all())    # the winner is always in its top group
        self.assertFalse(in_top_group(m, X, []).any())


class TestChurn(unittest.TestCase):
    def test_prediction_churn(self):
        out = prediction_churn([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(out, {"changed_share": 0.5, "fixed": 1, "broken": 1, "changed": 2})

    def test_explanation_churn_and_top_k(self):
        s = np.array([[3.0, 2.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        self.assertEqual(top_k_sets(s, 2), [frozenset({0, 1}), frozenset({2, 3})])
        self.assertEqual(top_k_sets(np.zeros((1, 4)), 2), [frozenset({0, 1})])   # stable on ties
        same = explanation_churn(s, s.copy(), k=2)
        self.assertEqual((same["any_change_share"], same["topk_change_share"], same["n"]), (0.0, 0.0, 2))
        t = s.copy()
        t[0, 2] = 5.0                                      # row 0's top 2 becomes {2, 0}
        t[1, 0] = 0.5                                      # row 1 changes, top 2 stays {2, 3}
        out = explanation_churn(s, t, k=2)
        self.assertEqual((out["any_change_share"], out["topk_change_share"]), (1.0, 0.5))
        self.assertEqual(explanation_churn(np.zeros((0, 3)), np.zeros((0, 3)))["n"], 0)


if __name__ == "__main__":
    unittest.main()
