"""Unit tests for Trust Diagnostics, Hypothesis Testing, and Caching modules."""

import unittest
import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.caching import benchmark_caching_modes
from pace_cfmnn.trust_diagnostics import (
    compute_diagnostics,
    compute_pd,
    compute_u,
    test_central_hypothesis,
)


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(999)
        n, d = 150, 3
        X = rng.uniform(size=(n, d))
        y = (X[:, 0] > 0.5).astype(int)
        self.model = CFMNN().fit(X, y)
        self.X = X
        self.y = y

    def test_compute_u_bounds(self):
        u = compute_u(self.model, self.X)
        self.assertEqual(len(u), len(self.X))
        self.assertTrue((u >= 0.0).all())
        self.assertTrue((u <= 1.0).all())

    def test_compute_pd_bounds(self):
        pd = compute_pd(self.model, self.X, K=20, radius=0.05)
        self.assertEqual(len(pd), len(self.X))
        self.assertTrue((pd >= 0.0).all())
        self.assertTrue((pd <= 1.0).all())

    def test_compute_diagnostics_and_hypothesis(self):
        split = 100
        X_tr, y_tr = self.X[:split], self.y[:split]
        X_te, y_te = self.X[split:], self.y[split:]

        sig_tr = compute_diagnostics(self.model, X_tr, background=X_tr[:30], K=15, radius=0.05)
        sig_te = compute_diagnostics(self.model, X_te, background=X_tr[:30], K=15, radius=0.05)

        y_tr_err = (self.model.predict(X_tr) != y_tr).astype(int)
        y_te_err = (self.model.predict(X_te) != y_te).astype(int)

        # Ensure at least one error for testing metrics
        y_te_err[0] = 1
        y_te_err[1] = 0

        res = test_central_hypothesis(sig_tr, y_tr_err, sig_te, y_te_err, n_bootstrap=100)
        self.assertFalse(np.isnan(res.auroc_base))
        self.assertFalse(np.isnan(res.auroc_ext))
        self.assertIsInstance(res.delta_auroc, float)


class TestCaching(unittest.TestCase):
    def test_caching_benchmark(self):
        rng = np.random.default_rng(123)
        X = rng.uniform(size=(80, 3))
        y = (X[:, 0] > 0.5).astype(int)
        m = CFMNN().fit(X, y)

        results = benchmark_caching_modes(m, X[:20], background=X[:20], rho=0.02, K=10)
        self.assertEqual(len(results), 3)
        modes = [r.mode for r in results]
        self.assertIn("Exact Structural Cache", modes)
        self.assertIn("Approximate Cache", modes)

        # Exact cache must have exactly 0.0 structural mismatch rate!
        exact_res = [r for r in results if "Exact" in r.mode][0]
        self.assertEqual(exact_res.structural_mismatch_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
