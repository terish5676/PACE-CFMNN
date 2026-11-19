"""Smoke tests: baselines run on the same model and return the right shapes."""

import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn import xai_baselines as xb


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.uniform(size=(300, 4))
        cls.y = (cls.X[:, 0] > 0.5).astype(int)
        cls.m = CFMNN().fit(cls.X, cls.y)

    def test_scores_are_distributions(self):
        s = xb.class_scores(self.m, self.X)
        np.testing.assert_allclose(s.sum(axis=1), 1.0)

    def test_kernel_shap_and_lime_shapes(self):
        a = xb.kernel_shap(self.m, self.X[:3], self.X[:50], nsamples=50)
        b = xb.lime_tabular(self.m, self.X[:3], self.X, num_samples=200)
        self.assertEqual(a.shape, (3, 4))
        self.assertEqual(b.shape, (3, 4))
        # Feature 0 decides the class, so both should rank it first on average.
        self.assertEqual(np.abs(a).mean(axis=0).argmax(), 0)
        self.assertEqual(np.abs(b).mean(axis=0).argmax(), 0)

    def test_kernel_shap_repeatable_and_leaves_global_rng_alone(self):
        # 10 features: 1,022 coalitions > nsamples, so shap must sample at random
        # (with 4 features it enumerates all 14 and no randomness is involved).
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(300, 10))
        m = CFMNN().fit(X, (X[:, 0] + X[:, 1] > 1).astype(int))
        before = np.random.get_state()[1].copy()
        a = xb.kernel_shap(m, X[:3], X[:50], nsamples=50, seed=7)
        np.testing.assert_array_equal(before, np.random.get_state()[1])
        np.random.uniform(size=100)                       # disturb the global state
        b = xb.kernel_shap(m, X[:3], X[:50], nsamples=50, seed=7)
        np.testing.assert_array_equal(a, b)
        c = xb.kernel_shap(m, X[:3], X[:50], nsamples=50, seed=8)
        self.assertFalse(np.array_equal(a, c))           # another seed, another sample

    def test_occlusion_equals_pace_flip(self):
        from pace_cfmnn.explain import pace_flip
        a = xb.occlusion(self.m, self.X[:15], self.X[100:])
        b = pace_flip(self.m, self.X[:15], self.X[100:])
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_occlusion_batched_equals_loop(self):
        X, bg = self.X[:15], self.X[100:]
        a = xb.occlusion(self.m, X, bg)
        np.testing.assert_allclose(xb.occlusion_batched(self.m, X, bg), a, rtol=0, atol=1e-9)
        # one point per model call, so the batches are split
        np.testing.assert_allclose(xb.occlusion_batched(self.m, X, bg, max_rows=1), a,
                                   rtol=0, atol=1e-9)
        vals = np.random.default_rng(0).uniform(size=(15, 8, X.shape[1]))  # own values per point
        np.testing.assert_allclose(xb.occlusion_batched(self.m, X, None, values=vals),
                                   xb.occlusion(self.m, X, None, values=vals), rtol=0, atol=1e-9)

    def test_nun_and_dice(self):
        x = self.X[0]
        t = 1 - self.m.predict(x[None])[0]
        nun = xb.nearest_unlike_neighbour(self.m, x, self.X, t)
        self.assertEqual(self.m.predict(nun[None])[0], t)
        cf = xb.dice_counterfactual(self.m, x, self.X, self.y, t)
        if cf is not None:
            self.assertEqual(self.m.predict(cf[None])[0], t)


if __name__ == "__main__":
    unittest.main()


class TestFairProtocol(unittest.TestCase):
    """The v2 RQ1 protocol: margin targets, signed ranking, insertion, fairer DiCE."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.X = rng.uniform(size=(400, 4))
        cls.y = (cls.X[:, 0] + 0.3 * cls.X[:, 1] > 0.65).astype(int)
        cls.m = CFMNN().fit(cls.X, cls.y)

    def test_margin_score_matches_supports(self):
        det = self.m.predict_details(self.X[:20])
        s = det.support_rescaled
        a = det.cls
        got = xb.margin_score(self.m, self.X[:20], 1)
        np.testing.assert_allclose(got, s[:, 1] - s[:, 0])
        self.assertTrue((xb.margin_score(self.m, self.X[:20][a == 0], 0) >= -1e-12).all())

    def test_kernel_shap_margin_is_efficient(self):
        # With 4 features shap enumerates every coalition, so the values add up exactly
        # to f(x) minus the mean of f over the background (Shapley efficiency).
        bg = self.X[:30]
        x = self.X[100:105]
        phi = xb.kernel_shap(self.m, x, bg, nsamples=500, target="margin")
        cls = self.m.predict_details(x).cls
        for r in range(len(x)):
            f = lambda Z: xb.margin_score(self.m, Z, cls[r])
            self.assertAlmostEqual(phi[r].sum(), f(x[r:r + 1])[0] - f(bg).mean(), places=6)

    def test_lime_margin_returns_contributions(self):
        b = xb.lime_tabular(self.m, self.X[:3], self.X, num_samples=300, target="margin")
        self.assertEqual(b.shape, (3, 4))
        self.assertTrue(np.isfinite(b).all())

    def test_signed_ranking_and_insertion(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_faithfulness as rf
        X, bg = self.X[200:240], self.X[:100]
        good = np.abs(xb.occlusion(self.m, X, bg))           # never negative
        _, signed = rf.deletion_auc(self.m, X, good, bg, signed=True)
        _, absolute = rf.deletion_auc(self.m, X, good, bg, signed=False)
        np.testing.assert_array_equal(signed, absolute)       # same order when nothing is negative
        flipped = -good                                       # most supportive feature last
        self.assertGreater(rf.deletion_auc(self.m, X, flipped, bg, signed=True)[1].mean(),
                           signed.mean())
        _, ins_good = rf.insertion_auc(self.m, X, good, bg)
        _, ins_bad = rf.insertion_auc(self.m, X, flipped, bg)
        self.assertGreater(ins_good.mean(), ins_bad.mean())  # higher = more faithful
        t3 = rf.top3_random_ties(np.zeros((50, 4)), np.random.default_rng(0))
        self.assertGreater(len({tuple(sorted(r)) for r in t3}), 1)   # ties not broken by index

    def test_dice_nearest_valid(self):
        x = self.X[self.m.predict(self.X) == 0][0]
        cf = xb.dice_counterfactual(self.m, x, self.X, self.y, 1, seed=0, n_cfs=4, sparse=True,
                                    pick="nearest_valid")
        if cf is not None:
            self.assertEqual(self.m.predict(cf[None])[0], 1)
