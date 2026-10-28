"""Smoke tests: baselines run on the same model and return the right shapes."""

import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn import xai_baselines as xb


try:
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    HAS_DICE = True
except ImportError:
    HAS_DICE = False


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

    @unittest.skipUnless(HAS_SHAP, "shap not installed")
    def test_kernel_shap_shapes(self):
        a = xb.kernel_shap(self.m, self.X[:3], self.X[:50], nsamples=50)
        self.assertEqual(a.shape, (3, 4))
        # Feature 0 decides the class, so it should rank first on average.
        self.assertEqual(np.abs(a).mean(axis=0).argmax(), 0)

    @unittest.skipUnless(HAS_LIME, "lime not installed")
    def test_lime_shapes(self):
        b = xb.lime_tabular(self.m, self.X[:3], self.X, num_samples=200)
        self.assertEqual(b.shape, (3, 4))
        # Feature 0 decides the class, so it should rank first on average.
        self.assertEqual(np.abs(b).mean(axis=0).argmax(), 0)

    @unittest.skipUnless(HAS_SHAP, "shap not installed")
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

    def test_nearest_unlike_neighbour(self):
        x = self.X[0]
        t = 1 - self.m.predict(x[None])[0]
        nun = xb.nearest_unlike_neighbour(self.m, x, self.X, t)
        self.assertEqual(self.m.predict(nun[None])[0], t)

    @unittest.skipUnless(HAS_DICE, "dice_ml not installed")
    def test_dice_counterfactual(self):
        x = self.X[0]
        t = 1 - self.m.predict(x[None])[0]
        cf = xb.dice_counterfactual(self.m, x, self.X, self.y, t)
        if cf is not None:
            self.assertEqual(self.m.predict(cf[None])[0], t)


if __name__ == "__main__":
    unittest.main()
