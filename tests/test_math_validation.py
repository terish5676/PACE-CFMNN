"""Mathematical Validation Suite (Protocol v6.0 Appendix A & B).

Validates:
1. Continuity of the implemented membership ambiguity score U(x):
   - Tests continuity across decision boundaries, hyperbox boundaries, and score-tie points.
   - Numerically validates lim_{delta -> 0} |U(x + delta) - U(x)| = 0.
2. Exactness of the Lipschitz constant and sufficient condition for caching:
   - Numerically verifies that |m_j(x + delta) - m_j(x)| <= (gamma / 2) * ||delta||_inf
     across dense random samples.
   - Verifies that whenever m_{j*}(x) - m_j(x) > gamma * rho, j* remains strictly
     ahead for 100% of tested perturbations inside the L_inf ball of radius rho.
"""

import unittest
import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.caching import (
    compute_exact_lipschitz_constant,
    verify_winning_box_invariance_condition,
)
from pace_cfmnn.trust_diagnostics import compute_u


class TestMathematicalContinuityU(unittest.TestCase):
    """Rigorous numerical validation of Appendix A continuity theorem."""

    def setUp(self):
        rng = np.random.default_rng(1234)
        n, d = 400, 4
        X = rng.uniform(size=(n, d))
        y = ((X[:, 0] + X[:, 1] > 1.0)).astype(int)
        self.model = CFMNN(theta=0.3, gamma=4.0).fit(X, y)
        self.rng = rng

    def test_continuity_under_vanishing_perturbations(self):
        """Test that |U(x + delta) - U(x)| -> 0 as ||delta|| -> 0 on 50 test points."""
        X_test = self.rng.uniform(size=(50, 4))
        u_base = compute_u(self.model, X_test, eta=0.5)

        deltas = [1e-2, 1e-3, 1e-4, 1e-5]
        diffs = []
        for eps in deltas:
            pert = np.clip(X_test + eps * self.rng.normal(size=X_test.shape), 0.0, 1.0)
            u_pert = compute_u(self.model, pert, eta=0.5)
            max_diff = np.max(np.abs(u_pert - u_base))
            diffs.append(max_diff)

        # Differences should strictly decrease as epsilon shrinks towards 0
        for i in range(len(diffs) - 1):
            self.assertLess(diffs[i + 1], diffs[i] + 1e-7)
        self.assertLess(diffs[-1], 1e-4)

    def test_continuity_at_exact_tie_points(self):
        """Continuity specifically at points where class scores tie (r1 == r2)."""
        # Find points where class support difference is tiny or construct one
        pts = self.rng.uniform(size=(200, 4))
        det = self.model.predict_details(pts)
        margin = np.abs(det.support[:, 0] - det.support[:, 1])
        closest_idx = np.argmin(margin)
        x_tie = pts[closest_idx]

        u_tie = compute_u(self.model, x_tie[None, :], eta=0.5)[0]
        # Perturb slightly in random directions
        for _ in range(20):
            delta = self.rng.uniform(-1e-4, 1e-4, size=4)
            x_pert = np.clip(x_tie + delta, 0.0, 1.0)
            u_pert = compute_u(self.model, x_pert[None, :], eta=0.5)[0]
            self.assertLess(abs(u_pert - u_tie), 5e-3)


class TestMathematicalLipschitzAndCaching(unittest.TestCase):
    """Rigorous verification of Appendix B exact-cache condition and Lipschitz constants."""

    def setUp(self):
        rng = np.random.default_rng(5678)
        n, d = 300, 4
        X = rng.uniform(size=(n, d))
        y = ((X[:, 0] * X[:, 1] > 0.25)).astype(int)
        self.gamma = 4.0
        self.model = CFMNN(theta=0.3, gamma=self.gamma).fit(X, y)
        self.rng = rng

    def test_lipschitz_constant_empirical_bound(self):
        """Verify |m_j(x + delta) - m_j(x)| <= (gamma / 2) * ||delta||_inf on 1000 pairs."""
        L_theoretical = compute_exact_lipschitz_constant(self.gamma)  # 2.0
        self.assertEqual(L_theoretical, 2.0)

        n_pairs = 1000
        x1 = self.rng.uniform(size=(n_pairs, 4))
        delta = self.rng.uniform(-0.1, 0.1, size=(n_pairs, 4))
        x2 = np.clip(x1 + delta, 0.0, 1.0)

        m1 = self.model.memberships(x1)
        m2 = self.model.memberships(x2)
        diff_m = np.abs(m2 - m1)  # shape (n_pairs, H)

        delta_inf = np.max(np.abs(x2 - x1), axis=1)[:, None]  # shape (n_pairs, 1)
        bound = L_theoretical * delta_inf + 1e-9

        # No difference may exceed theoretical Lipschitz bound
        violations = np.sum(diff_m > bound)
        self.assertEqual(violations, 0, f"Found {violations} Lipschitz bound violations!")

    def test_exact_caching_sufficient_condition_guarantee(self):
        """Verify that when condition holds, j* never changes in 100% of tested perturbations."""
        rho = 0.02
        X_test = self.rng.uniform(size=(100, 4))
        is_provable, winner, margin = verify_winning_box_invariance_condition(self.model, X_test, rho)

        # For all samples where invariance is proven, sample 50 perturbations inside B_inf(x, rho)
        provable_indices = np.flatnonzero(is_provable)
        self.assertGreater(len(provable_indices), 0, "Should have provably invariant samples.")

        for idx in provable_indices:
            x = X_test[idx]
            w_orig = winner[idx]
            pert = np.clip(x + self.rng.uniform(-rho, rho, size=(50, 4)), 0.0, 1.0)
            det_pert = self.model.predict_details(pert)
            # 100% of perturbations must keep winning box w_orig
            self.assertTrue(np.all(det_pert.winner == w_orig),
                            f"Exact cache condition violated for point {idx}!")


if __name__ == "__main__":
    unittest.main()
