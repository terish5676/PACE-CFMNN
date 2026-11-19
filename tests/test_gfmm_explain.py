"""Tough review B2: PACE-SHAP and PACE-flip for min-membership (GFMM-family) models."""

import importlib.util
import unittest
import warnings

import numpy as np

from pace_cfmnn.gfmm_explain import (GFMMBoxes, PaceFlipMin, PaceShapMin, brute_force_min,
                                     subset_mins)
from tests.test_refine import problem

HAS_HB = importlib.util.find_spec("hbbrain") is not None


def fit_gfmm(seed, d, kind="online"):
    X, y = problem(seed, n=300, d=d, classes=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "online":
            from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
            est = OnlineGFMM(theta=0.3, theta_min=0.3, gamma=4).fit(X, y)
        else:
            from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
            est = ImprovedOnlineGFMM(theta=0.3, gamma=4).fit(X, y)
    return est, X


class TestSubsetMins(unittest.TestCase):
    def test_against_direct(self):
        rng = np.random.default_rng(0)
        T = rng.uniform(size=(3, 5))
        out = subset_mins(T)
        for s in range(32):
            idx = [i for i in range(5) if s >> i & 1]
            want = T[:, idx].min(axis=1) if idx else np.full(3, np.inf)
            np.testing.assert_array_equal(out[:, s], want)


@unittest.skipUnless(HAS_HB, "hyperbox-brain not installed")
class TestGFMM(unittest.TestCase):
    def test_adapter_predicts_like_hyperbox_brain(self):
        for kind in ("online", "iol"):
            est, X = fit_gfmm(0, 4, kind)
            bx = GFMMBoxes.from_hbbrain(est)
            rng = np.random.default_rng(1)
            P = np.vstack([X, rng.uniform(size=(200, 4))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                want = est.predict(P)
            np.testing.assert_array_equal(bx.predict(P), want)

    def test_iol_gfmm_ties_follow_its_probability_rule(self):
        """P4 Amendment 1: IOL-GFMM breaks ties by its sample-count rule, not by Manhattan distance.

        Class 0 has many rows near the origin, class 1 a few rows nearer (1, 1). At (1, 1) every
        box has membership 0, so all classes tie: the nearest centre is class 1, but IOL-GFMM
        picks the class whose tied boxes hold more rows, class 0.
        """
        from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
        rng = np.random.default_rng(5)
        X = np.vstack([rng.uniform(0.0, 0.3, size=(120, 2)), rng.uniform(0.35, 0.45, size=(6, 2))])
        y = np.r_[np.zeros(120, int), np.ones(6, int)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = ImprovedOnlineGFMM(theta=0.3, gamma=4).fit(X, y)
            P = np.vstack([[[1.0, 1.0], [0.95, 0.9], [1.0, 0.8]], rng.uniform(size=(200, 2))])
            want = est.predict(P)
        bx = GFMMBoxes.from_hbbrain(est)
        self.assertTrue((bx.memberships(P[:3]) == 0).all())          # full ties at membership 0
        manhattan = GFMMBoxes(est.V, est.W, est.C, est.gamma)         # the old rule decides otherwise
        self.assertTrue((manhattan.predict(P[:3]) != want[:3]).all())
        np.testing.assert_array_equal(bx.predict(P), want)

    def test_exact_values_equal_brute_force(self):
        for kind, seed, d in (("online", 0, 3), ("online", 1, 5), ("iol", 2, 6)):
            est, X = fit_gfmm(seed, d, kind)
            bx = GFMMBoxes.from_hbbrain(est)
            rng = np.random.default_rng(seed)
            bg = X[rng.choice(len(X), 12, replace=False)]
            pts = np.vstack([X[rng.choice(len(X), 4, replace=False)], rng.uniform(size=(2, d))])
            for prune in (True, False):
                ps = PaceShapMin(bx, bg, prune=prune)
                phi, fx, f0 = ps.explain(pts, return_parts=True)
                np.testing.assert_allclose(phi.sum(axis=1), fx - f0, atol=1e-12)
                for p in range(len(pts)):
                    np.testing.assert_allclose(phi[p], brute_force_min(bx, pts[p], bg),
                                               rtol=0, atol=1e-12)

    def test_sampled_mode(self):
        est, X = fit_gfmm(3, 5)
        bx = GFMMBoxes.from_hbbrain(est)
        rng = np.random.default_rng(3)
        bg = X[rng.choice(len(X), 8, replace=False)]
        pts = X[rng.choice(len(X), 3, replace=False)]
        exact = PaceShapMin(bx, bg).explain(pts)
        samp, fx, f0 = PaceShapMin(bx, bg, exact_max_d=2, n_permutations=400).explain(
            pts, return_parts=True)
        np.testing.assert_allclose(samp.sum(axis=1), fx - f0, atol=1e-12)
        self.assertLess(np.abs(samp - exact).max(), 0.03)

    def test_pace_flip_min_equals_occlusion(self):
        est, X = fit_gfmm(4, 4)
        bx = GFMMBoxes.from_hbbrain(est)
        rng = np.random.default_rng(4)
        vals = np.quantile(X, (np.arange(16) + 0.5) / 16, axis=0)
        pts = np.vstack([X[rng.choice(len(X), 5, replace=False)], rng.uniform(size=(3, 4))])
        _, flip, _ = PaceFlipMin(bx, vals).explain(pts, return_parts=True)
        base = bx.predict_index(pts)
        for p in range(len(pts)):
            for i in range(4):
                P = np.repeat(pts[p][None], len(vals), axis=0)
                P[:, i] = vals[:, i]
                self.assertAlmostEqual(flip[p, i], np.mean(bx.predict_index(P) != base[p]))


if __name__ == "__main__":
    unittest.main()
