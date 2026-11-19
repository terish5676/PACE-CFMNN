"""Tough review B3, B8 and B9: decision explanation, exact flip sets, faster exact mode."""

import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.decision import PaceDecision, brute_force_decision
from pace_cfmnn.explain import PaceShap, brute_force_shap, subset_sums
from pace_cfmnn.flip_exact import flip_sets
from tests.test_refine import problem


def fitted(seed=0, n=400, d=5, classes=3, theta=0.3):
    X, y = problem(seed, n=n, d=d, classes=classes)
    return CFMNN(theta=theta).fit(X, y), X


class TestRecurrence(unittest.TestCase):
    def test_subset_sums(self):
        rng = np.random.default_rng(0)
        base, D = rng.normal(size=3), rng.normal(size=(3, 6))
        masks = np.arange(64)
        ind = ((masks[None, :] >> np.arange(6)[:, None]) & 1).astype(float)
        np.testing.assert_allclose(subset_sums(base, D), base[:, None] + D @ ind, atol=1e-13)

    def test_same_values_as_matmul_and_brute_force(self):
        for seed, d in ((0, 4), (1, 6), (2, 8)):
            m, X = fitted(seed, d=d)
            rng = np.random.default_rng(seed)
            bg = X[rng.choice(len(X), 15, replace=False)]
            pts = rng.uniform(size=(6, d))
            a = PaceShap(m, bg).explain(pts)
            b = PaceShap(m, bg, exact_algo="recurrence").explain(pts)
            c = brute_force_shap(m, pts, bg)
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)
            np.testing.assert_allclose(c, a, rtol=0, atol=1e-12)

    def test_recurrence_above_twelve_features(self):
        m, X = fitted(3, n=300, d=13, classes=2)
        rng = np.random.default_rng(3)
        bg = X[rng.choice(len(X), 4, replace=False)]
        pts = rng.uniform(size=(2, 13))
        ps = PaceShap(m, bg, exact_max_d=13, exact_algo="recurrence")
        phi, fx, f0 = ps.explain(pts, return_parts=True)
        self.assertEqual(ps.stats_["mode"], "exact")
        np.testing.assert_allclose(phi.sum(axis=1), fx - f0, atol=1e-12)
        np.testing.assert_allclose(phi, brute_force_shap(m, pts, bg), rtol=0, atol=1e-12)

    def test_bad_algo(self):
        m, X = fitted()
        with self.assertRaises(ValueError):
            PaceShap(m, X[:3], exact_algo="fast")


class TestFlipSets(unittest.TestCase):
    def test_class_on_a_dense_grid_matches_the_pieces(self):
        grid = np.linspace(0.0, 1.0, 2001)
        for seed, d in ((0, 3), (1, 5)):
            m, X = fitted(seed, d=d)
            rng = np.random.default_rng(seed)
            for x in np.vstack([X[rng.choice(len(X), 5, replace=False)], rng.uniform(size=(5, d))]):
                for fs in flip_sets(m, x):
                    P = np.repeat(x[None], len(grid), axis=0)
                    P[:, fs.feature] = grid
                    cl = m.predict_details(P).cls
                    k = np.searchsorted(fs.breaks, grid, side="right") - 1
                    on_break = np.isclose(grid[:, None], fs.breaks[None, :], atol=1e-9).any(axis=1)
                    k = np.clip(k, 0, len(fs.open_cls) - 1)
                    expect = fs.open_cls[k]
                    ok = on_break | (cl == expect)
                    self.assertTrue(ok.all(), (seed, fs.feature))

    def test_smallest_change_and_intervals(self):
        m, X = fitted(0, d=3)
        rng = np.random.default_rng(5)
        found = 0
        for x in X[rng.choice(len(X), 20, replace=False)]:
            for fs in flip_sets(m, x):
                dist, _ = fs.smallest_change(x[fs.feature])
                iv = fs.flip_intervals()
                if not iv:
                    self.assertEqual(dist, np.inf)
                    continue
                found += 1
                near = min(0.0 if lo <= x[fs.feature] <= hi else min(abs(lo - x[fs.feature]),
                           abs(hi - x[fs.feature])) for lo, hi in iv)
                self.assertAlmostEqual(dist, near, places=12)
                # a value just inside the nearest flipping piece flips the decision
                lo, hi = min(iv, key=lambda t: min(abs(t[0] - x[fs.feature]), abs(t[1] - x[fs.feature])))
                u = 0.5 * (lo + hi)
                p = x.copy()
                p[fs.feature] = u
                if hi > lo:
                    self.assertNotEqual(int(m.predict_details(p[None]).cls[0]), fs.cls_x)
        self.assertGreater(found, 0)


class TestPaceDecision(unittest.TestCase):
    def test_exact_equals_brute_force(self):
        for seed, d in ((0, 3), (1, 5), (2, 6)):
            m, X = fitted(seed, d=d)
            rng = np.random.default_rng(seed)
            bg = X[rng.choice(len(X), 12, replace=False)]
            pts = np.vstack([X[rng.choice(len(X), 4, replace=False)], rng.uniform(size=(2, d))])
            pd_ = PaceDecision(m, bg)
            phi, vN, v0 = pd_.explain(pts, return_parts=True)
            np.testing.assert_allclose(vN, 1.0)
            np.testing.assert_allclose(phi.sum(axis=1), vN - v0, atol=1e-12)
            for p in range(len(pts)):
                np.testing.assert_allclose(phi[p], brute_force_decision(m, pts[p], bg),
                                           rtol=0, atol=1e-12)

    def test_sampled_mode_adds_up_and_is_close(self):
        m, X = fitted(4, d=5)
        rng = np.random.default_rng(4)
        bg = X[rng.choice(len(X), 10, replace=False)]
        pts = X[rng.choice(len(X), 3, replace=False)]
        exact = PaceDecision(m, bg).explain(pts)
        samp, vN, v0 = PaceDecision(m, bg, exact_max_d=2, n_permutations=400).explain(
            pts, return_parts=True)
        np.testing.assert_allclose(samp.sum(axis=1), vN - v0, atol=1e-12)
        self.assertLess(np.abs(samp - exact).max(), 0.05)

    def test_ties_get_nonzero_values(self):
        """At a tied training row the margin is 0, but the decision game still credits features."""
        m, X = fitted(0, d=4)
        det = m.predict_details(X)
        tied = np.flatnonzero(det.n_tied > 1)
        self.assertGreater(len(tied), 0)
        bg = X[:10]
        phi = PaceDecision(m, bg).explain(X[tied[:3]])
        self.assertTrue((np.abs(phi).sum(axis=1) > 0).any())


if __name__ == "__main__":
    unittest.main()
