"""PACE-SHAP: exact Shapley values of the predicted class's margin."""

import importlib.util
import unittest

import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.explain import PaceShap, pace_shap
from pace_cfmnn.xai_baselines import kernel_shap, margin_score
from tests.test_explain import exact_shapley, problem


def brute(m, X, bg):
    """Brute-force Shapley of the margin, re-running the model on every coalition."""
    cls = m.predict_details(X).cls
    return np.array([exact_shapley(lambda Z, a=a: margin_score(m, Z, a), x, bg)
                     for x, a in zip(X, cls)])


class TestPaceShapExact(unittest.TestCase):
    def test_equals_brute_force(self):
        k = 0
        for d, classes in [(2, 2), (4, 3), (6, 4)]:
            for grid in (None, 2):
                with self.subTest(d=d, classes=classes, grid=grid):
                    X, y = problem(20 + k, d=d, classes=classes, grid=grid, label_noise=0.1)
                    m = CFMNN().fit(X[:250], y[:250])
                    Xe, bg = X[250:258], X[:9]
                    np.testing.assert_allclose(PaceShap(m, bg).explain(Xe), brute(m, Xe, bg),
                                               atol=1e-12)
                k += 1

    def test_single_class_model(self):
        X, _ = problem(30)
        m = CFMNN().fit(X[:200], np.zeros(200, dtype=int))
        Xe, bg = X[200:206], X[:8]
        phi, fx, f0 = PaceShap(m, bg).explain(Xe, return_parts=True)
        # No other class: margin_score is infinite, so PACE-SHAP explains s'_A itself.
        own = lambda Z: m.predict_details(Z).support_rescaled[:, 0]
        np.testing.assert_allclose(phi, [exact_shapley(own, x, bg) for x in Xe], atol=1e-12)
        np.testing.assert_allclose(phi.sum(1), fx - f0, atol=1e-12)

    def test_pruning_and_chunks_change_nothing(self):
        X, y = problem(31, d=6, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe, bg = X[250:270], X[:40]
        ref = PaceShap(m, bg, prune=False).explain(Xe)
        for chunk in (64, 10**9):
            for exact_max_d in (12, 0):
                ex = PaceShap(m, bg, chunk_elems=chunk, exact_max_d=exact_max_d, seed=3)
                out = ex.explain(Xe)
                if exact_max_d:
                    np.testing.assert_allclose(out, ref, atol=1e-12)
                    self.assertGreater(ex.stats_["share_of_boxes_skipped"], 0)
                else:
                    raw = PaceShap(m, bg, chunk_elems=chunk, exact_max_d=0, seed=3,
                                   prune=False).explain(Xe)
                    np.testing.assert_allclose(out, raw, atol=1e-12)


class TestPaceShapProperties(unittest.TestCase):
    def setUp(self):
        X, y = problem(40, d=5, classes=3)
        self.m = CFMNN().fit(X[:250], y[:250])
        self.Xe, self.bg = X[250:265], X[:30]

    def test_efficiency_and_value(self):
        cls = self.m.predict_details(self.Xe).cls
        margin = np.array([margin_score(self.m, x[None], a)[0] for x, a in zip(self.Xe, cls)])
        base = np.array([margin_score(self.m, self.bg, a).mean() for a in cls])
        for exact_max_d in (12, 0):
            phi, fx, f0 = PaceShap(self.m, self.bg, exact_max_d=exact_max_d).explain(
                self.Xe, return_parts=True)
            np.testing.assert_allclose(fx, margin, atol=1e-12)
            np.testing.assert_allclose(f0, base, atol=1e-12)
            np.testing.assert_allclose(phi.sum(1), fx - f0, atol=1e-12)

    def test_dummy_feature_is_zero(self):
        x = self.Xe[:1]
        bg = self.bg.copy()
        bg[:, 2] = x[0, 2]             # feature 2 never differs from x: it cannot matter
        for exact_max_d in (12, 0):
            phi = PaceShap(self.m, bg, exact_max_d=exact_max_d).explain(x)
            self.assertAlmostEqual(phi[0, 2], 0.0, places=12)

    def test_sampling_converges_and_is_seeded(self):
        exact = PaceShap(self.m, self.bg).explain(self.Xe)
        ex = PaceShap(self.m, self.bg, exact_max_d=0, n_permutations=400, seed=1)
        a = ex.explain(self.Xe)
        np.testing.assert_allclose(a, exact, atol=0.02 * np.abs(exact).max())
        np.testing.assert_array_equal(a, ex.explain(self.Xe, seed=1))
        self.assertFalse(np.array_equal(a, ex.explain(self.Xe, seed=2)))
        # The exact mode does not depend on the seed at all.
        np.testing.assert_array_equal(exact, PaceShap(self.m, self.bg, seed=9).explain(self.Xe))

    def test_two_features_antithetic_is_exact(self):
        # With d = 2 an order and its reverse are all the orders there are.
        X, y = problem(41, d=2, classes=2)
        m = CFMNN().fit(X[:200], y[:200])
        Xe, bg = X[200:210], X[:20]
        np.testing.assert_allclose(PaceShap(m, bg, exact_max_d=0, n_permutations=2).explain(Xe),
                                   PaceShap(m, bg).explain(Xe), atol=1e-12)

    def test_rebuilds_when_boxes_change(self):
        ex = PaceShap(self.m, self.bg)
        ex.explain(self.Xe)
        self.m.W_[0] = np.minimum(self.m.W_[0] + 0.2, 1.0)
        np.testing.assert_allclose(ex.explain(self.Xe), PaceShap(self.m, self.bg).explain(self.Xe),
                                   atol=1e-12)
        np.testing.assert_allclose(pace_shap(self.m, self.Xe[:3], self.bg),
                                   brute(self.m, self.Xe[:3], self.bg), atol=1e-12)

    def test_rejects_odd_permutation_count(self):
        with self.assertRaises(ValueError):
            PaceShap(self.m, self.bg, n_permutations=3)


@unittest.skipUnless(importlib.util.find_spec("shap"), "shap not installed")
class TestAgainstKernelShap(unittest.TestCase):
    def test_equals_exhaustive_kernel_shap_on_same_rows(self):
        # With nsamples >= 2^d - 2, KernelSHAP enumerates every coalition and is
        # exact; on the same background rows it must equal PACE-SHAP.
        X, y = problem(50, d=4, classes=3)
        m = CFMNN().fit(X[:250], y[:250])
        Xe, bg = X[250:258], X[:12]
        ks = kernel_shap(m, Xe, bg, nsamples=2048, seed=0, target="margin", l1_reg=False)
        np.testing.assert_allclose(PaceShap(m, bg).explain(Xe), ks, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
