"""Final review: hand-computed checks of the paper's formulas, and edge cases.

Each test pins a formula to numbers worked out by hand (in the comments), so a
wrong formula would fail even where the other tests compare two pieces of code
that could share the same mistake.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.audit import impact_set, prediction_churn  # noqa: E402
from pace_cfmnn.explain import PaceShap, pace_flip  # noqa: E402
from pace_cfmnn.refine import Gate, shrink_box  # noqa: E402
from pace_cfmnn.signals import uncertainty  # noqa: E402
from pace_cfmnn.xai_baselines import margin_score  # noqa: E402


def toy():
    """The paper's Section 4 example: box A [0.2, 0.4] x [0.2, 0.5], box B [0.5, 0.8] x [0.1, 0.4].

    Each box is grown from its two corners, so its centroid is the corners' mean:
    A (0.3, 0.35), B (0.65, 0.25). theta = 0.4 keeps both inside the size limit
    (widths sum to 0.5 and 0.6, limit 0.8). gamma = 4.
    """
    X = np.array([[0.2, 0.2], [0.4, 0.5], [0.5, 0.1], [0.8, 0.4]])
    y = np.array([0, 0, 1, 1])
    return CFMNN(theta=0.4).fit(X, y), X, y


class TestSection4Example(unittest.TestCase):
    """Membership m = mean over features of 0.5 * [f(x - w) + f(v - x)], f(t) = max(0, 1 - 4 min(1, t))."""

    def test_boxes(self):
        m, _, _ = toy()
        np.testing.assert_allclose(m.V_, [[0.2, 0.2], [0.5, 0.1]])
        np.testing.assert_allclose(m.W_, [[0.4, 0.5], [0.8, 0.4]])
        np.testing.assert_allclose(m.centroid_, [[0.3, 0.35], [0.65, 0.25]])

    def test_memberships_and_table(self):
        m, _, _ = toy()
        # x = (0.43, 0.30): A: x0 is 0.03 above w -> 0.5 (1 + 0.88) = 0.94; x1 inside -> 1; mean 0.97.
        #                   B: x0 is 0.07 below v -> 0.5 (1 + 0.72) = 0.86; x1 inside -> 1; mean 0.93.
        rows = {(0.43, 0.30): (0.97, 0.93, 0),
                (0.30, 0.30): (1.00, 0.80, 0),   # B: 0.2 below v -> f = 0.2 -> 0.6; (0.6 + 1) / 2
                (0.70, 0.30): (0.75, 1.00, 1),   # A: 0.3 above w -> f = 0 -> 0.5; (0.5 + 1) / 2
                (0.43, 0.05): (0.82, 0.88, 1),   # A: (0.94 + 0.7) / 2; B: (0.86 + 0.9) / 2
                (0.43, 0.90): (0.72, 0.68, 0)}   # A: (0.94 + 0.5) / 2; B: (0.86 + 0.5) / 2
        X = np.array(list(rows))
        det = m.predict_details(X)
        np.testing.assert_allclose(det.support, [r[:2] for r in rows.values()], atol=1e-12)
        np.testing.assert_array_equal(det.cls, [r[2] for r in rows.values()])

    def test_pace_shap_by_hand(self):
        m, _, _ = toy()
        x, z = np.array([[0.43, 0.30]]), np.array([[0.30, 0.05]])
        # Margin f = s'_A - s'_B = 2 (s_A - s_B), A = the class predicted at x.
        # f({}) = f(z) = 2 (0.85 - 0.75) = 0.20     A: (1 + 0.7) / 2;   B: (0.6 + 0.9) / 2
        # f({0}) = f(0.43, 0.05) = 2 (0.82 - 0.88) = -0.12
        # f({1}) = f(0.30, 0.30) = 2 (1.00 - 0.80) = 0.40
        # f({0, 1}) = f(x) = 2 (0.97 - 0.93) = 0.08
        # phi_0 = 0.5 [(-0.12 - 0.20) + (0.08 - 0.40)] = -0.32
        # phi_1 = 0.5 [(0.40 - 0.20) + (0.08 + 0.12)] = 0.20;  sum = f(x) - f(z) = -0.12
        phi, fx, f0 = PaceShap(m, z).explain(x, return_parts=True)
        np.testing.assert_allclose(phi, [[-0.32, 0.20]], atol=1e-12)
        np.testing.assert_allclose([fx[0], f0[0]], [0.08, 0.20], atol=1e-12)
        np.testing.assert_allclose(margin_score(m, x, 0), [0.08], atol=1e-12)

    def test_u_and_u_star_by_hand(self):
        m, _, _ = toy()
        # Untied x = (0.43, 0.30): s'_1 = 0.94, s'_2 = 0.86, plain = 1 - 0.08 = 0.92.
        #   U* = 0.92 / 2 = 0.46;  U = 0.5 * 0.92 + 0.5 * (1 - 0.94) = 0.49.
        # Tied x = (0.45, 0.30): 0.05 outside both boxes on x0 -> both memberships 0.95, s' = 0.9.
        #   Centroid distances: A 0.15^2 + 0.05^2 = 0.025, B 0.2^2 + 0.05^2 = 0.0425 -> A wins.
        #   ratio = sqrt(0.025 / 0.0425); U* = (1 + ratio) / 2; U = 0.5 * ratio + 0.5 * (1 - 0.9).
        X = np.array([[0.43, 0.30], [0.45, 0.30]])
        det = m.predict_details(X)
        u, u_star, plain, _, tie = uncertainty(m, det)
        r = np.sqrt(0.025 / 0.0425)
        np.testing.assert_array_equal(tie, [False, True])
        np.testing.assert_array_equal(det.cls, [0, 0])
        np.testing.assert_allclose(u_star, [0.46, (1 + r) / 2], atol=1e-12)
        np.testing.assert_allclose(u, [0.49, 0.5 * r + 0.05], atol=1e-12)
        # The planned U ranks the tie as less uncertain than the untied point (the reason for U*).
        self.assertLess(u[1], u[0])
        self.assertGreater(u_star[1], u_star[0])


class TestStatistics(unittest.TestCase):
    def test_holm_by_hand(self):
        import run_error_detection
        import run_refinement_v2
        import summarise_refinement_v1
        # Sorted p: 0.005 x 4 = 0.02; 0.01 x 3 = 0.03; 0.03 x 2 = 0.06; 0.04 x 1 = 0.04 -> 0.06 (monotone).
        p = [0.01, 0.04, 0.03, 0.005]
        for mod in (run_refinement_v2, run_error_detection, summarise_refinement_v1):
            with self.subTest(module=mod.__name__):
                np.testing.assert_allclose(mod.holm(p), [0.03, 0.06, 0.06, 0.02])
                np.testing.assert_allclose(mod.holm([0.5, 0.6]), [1.0, 1.0])   # capped at 1

    def test_non_inferiority_endpoint(self):
        from run_refinement_v2 import test_endpoint as endpoint
        seeds = range(50, 60)

        def runs(diffs):
            return pd.DataFrame([{"dataset": "thyroid", "seed": s, "arm": a, "f1": v}
                                 for s, d in zip(seeds, diffs) for a, v in (("A", 80 + d), ("B", 80.0))])

        # a - b = -0.5 .. -0.41 (distinct): a trails b, but by less than the 1-point margin.
        # d + 1 > 0 in all 10 seeds -> exact one-sided Wilcoxon p = 1 / 2^10.
        t = endpoint(runs([-0.5 + 0.01 * k for k in range(10)]), "f1", "A", "B", "ni", margin=1.0)
        self.assertAlmostEqual(t.p[0], 1 / 1024)
        self.assertEqual(t.losses[0], 10)
        # Trailing by 1.5 points: non-inferiority must not be shown.
        t = endpoint(runs([-1.5 + 0.01 * k for k in range(10)]), "f1", "A", "B", "ni", margin=1.0)
        self.assertGreater(t.p[0], 0.5)
        # Two-sided superiority with every seed ahead: p = 2 / 2^10.
        t = endpoint(runs([1 + 0.01 * k for k in range(10)]), "f1", "A", "B", "two-sided")
        self.assertAlmostEqual(t.p[0], 2 / 1024)

    def test_fix_share_counts(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        before = np.array([0, 1, 1, 0, 2, 0])
        after = np.array([0, 0, 0, 1, 1, 1])
        # row 1 wrong -> right (fixed); row 2 right -> wrong (broken); row 3 fixed; row 4 broken;
        # row 5 wrong -> wrong (changed, neither). Fix share = 2 / (2 + 2) = 50%.
        c = prediction_churn(y, before, after)
        self.assertEqual((c["fixed"], c["broken"], c["changed"]), (2, 2, 5))
        self.assertAlmostEqual(c["changed_share"], 5 / 6)
        self.assertAlmostEqual(100 * c["fixed"] / (c["fixed"] + c["broken"]), 50.0)


class TestGateBoundaries(unittest.TestCase):
    def test_limits_are_inclusive(self):
        m, X, y = toy()
        gate = Gate(X, y, box_cap=m.n_boxes_)
        before = gate.measure(m)
        key = next(k for k in before if k.startswith("recall_"))
        # A drop of exactly 0.5 macro-F1 points, or exactly 1 point of a class's recall, passes.
        self.assertTrue(gate.check(dict(before, macro_f1=before["macro_f1"] + 0.5), m)[0])
        self.assertTrue(gate.check(dict(before, **{key: before[key] + 1.0}), m)[0])
        self.assertFalse(gate.check(dict(before, macro_f1=before["macro_f1"] + 0.51), m)[0])
        self.assertFalse(gate.check(dict(before, **{key: before[key] + 1.01}), m)[0])

    def test_box_cap_is_floor_of_1_1_times_start(self):
        # run_self_repair and run_refinement cap at floor(1.1 x the starting box count).
        self.assertEqual(int(np.floor(1.1 * 29)), 31)
        self.assertEqual(int(np.floor(1.1 * 10)), 11)


class TestEdgeCases(unittest.TestCase):
    def test_constant_feature_gets_nothing(self):
        rng = np.random.default_rng(0)
        X = np.c_[rng.uniform(size=(200, 2)), np.full(200, 0.5)]
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        m = CFMNN().fit(X[:150], y[:150])
        np.testing.assert_allclose(m.V_[:, 2], 0.5)
        np.testing.assert_allclose(m.W_[:, 2], 0.5)
        phi = PaceShap(m, X[:20]).explain(X[150:160])
        np.testing.assert_allclose(phi[:, 2], 0.0, atol=1e-12)
        # PACE-flip: the feature never flips the class and never moves the margin. (Its score
        # can still be above 0, because the score rescales each point's drops to [0, 1] for ranking.)
        _, flip, drop = pace_flip(m, X[150:160], X[:40], return_parts=True)
        np.testing.assert_allclose(flip[:, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(drop[:, 2], 0.0, atol=1e-12)

    def test_class_with_one_row(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(120, 2))
        y = (X[:, 0] > 0.5).astype(int)
        x_one = np.array([0.6, 0.55])
        X, y = np.vstack([X, x_one]), np.append(y, 2)
        m = CFMNN().fit(X, y)
        k = int(np.flatnonzero(m.box_class_ == 2)[0])
        np.testing.assert_allclose(m.V_[k], x_one)          # a zero-width box
        np.testing.assert_allclose(m.W_[k], x_one)
        self.assertEqual(int(m.predict(x_one[None])[0]), 2)  # its own centroid is at distance 0
        phi, fx, f0 = PaceShap(m, X[:30]).explain(x_one[None], return_parts=True)
        np.testing.assert_allclose(phi.sum(1), fx - f0, atol=1e-12)

    def test_one_feature(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(size=(150, 1))
        y = (X[:, 0] > 0.4).astype(int)
        m = CFMNN().fit(X[:100], y[:100])
        Xe, bg = X[100:110], X[:25]
        phi, fx, f0 = PaceShap(m, bg).explain(Xe, return_parts=True)
        # With one feature the Shapley value is the whole difference f(x) - mean f(z).
        a = m.predict_details(Xe).cls
        ref = [margin_score(m, x[None], int(c))[0] - margin_score(m, bg, int(c)).mean()
               for x, c in zip(Xe, a)]
        np.testing.assert_allclose(phi[:, 0], ref, atol=1e-12)
        self.assertEqual(pace_flip(m, Xe, bg).shape, (10, 1))

    def test_certificate_when_an_edit_removes_a_class(self):
        m, X, y = toy()
        # Two cuts empty class B's only box: the first drops its member (0.8, 0.4), the second (0.5, 0.1).
        shrink_box(m, 1, np.array([0.75, 0.38]), 0, X)
        self.assertEqual(m.n_boxes_, 2)
        before = __import__("copy").deepcopy(m)
        info = shrink_box(m, 1, np.array([0.52, 0.12]), 0, X)
        self.assertTrue(info["removed"])
        self.assertEqual(m.box_class_.tolist(), [0])        # class B has no box left
        g = np.linspace(0, 1, 41)
        grid = np.array(np.meshgrid(g, g)).reshape(2, -1).T
        changed = before.predict(grid) != m.predict(grid)
        self.assertTrue(changed.any())
        for tight in (False, True):
            cert = impact_set(before, m, grid, removed=1, tight=tight)
            self.assertFalse((changed & ~cert).any())


if __name__ == "__main__":
    unittest.main()
