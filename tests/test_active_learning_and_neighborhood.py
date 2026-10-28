"""Unit tests for neighborhood, signals, pace_score, refine, and active learning."""

import unittest
import numpy as np

from pace_cfmnn import (
    CFMNN,
    detect_binary_columns,
    generate_perturbations,
    compute_uncertainty,
    compute_prediction_disagreement,
    compute_explanation_instability,
    compute_prs,
    RefinementConfig,
    RefinementResult,
    refine_sample,
    run_active_refinement,
    ActiveLearningResult,
)


class TestActiveLearningAndNeighborhood(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        # 3 continuous, 1 binary
        self.X = rng.uniform(0.0, 1.0, size=(120, 4))
        self.X[:, 3] = (self.X[:, 3] > 0.5).astype(float)
        self.y = (self.X[:, 0] + self.X[:, 1] > 1.0).astype(int)

        self.model = CFMNN(theta=0.3, gamma=4.0).fit(self.X[:60], self.y[:60])

    def test_detect_binary_columns(self):
        binary_mask = detect_binary_columns(self.X)
        self.assertEqual(len(binary_mask), 4)
        self.assertFalse(binary_mask[0])
        self.assertTrue(binary_mask[3])

    def test_generate_perturbations(self):
        pert = generate_perturbations(self.X[:10], n_perturb=15, sigma=0.05, seed=42)
        self.assertEqual(pert.shape, (10, 15, 4))
        self.assertTrue((pert >= 0.0).all() and (pert <= 1.0).all())

    def test_signals(self):
        X_sub = self.X[:15]
        u, _ = compute_uncertainty(self.model, X_sub)
        self.assertEqual(len(u), 15)
        self.assertTrue((u >= 0.0).all() and (u <= 1.0).all())

        pert = generate_perturbations(X_sub, n_perturb=10, sigma=0.05, seed=42)
        pd_vals = compute_prediction_disagreement(self.model, X_sub, pert)
        self.assertEqual(len(pd_vals), 15)
        self.assertTrue((pd_vals >= 0.0).all() and (pd_vals <= 1.0).all())

        ei_vals, _ = compute_explanation_instability(self.model, X_sub, pert)
        self.assertEqual(len(ei_vals), 15)
        self.assertTrue((ei_vals >= 0.0).all() and (ei_vals <= 1.0).all())

    def test_compute_prs(self):
        X_sub = self.X[:20]
        prs_res = compute_prs(self.model, X_sub, n_perturb=10, sigma=0.05, seed=42)
        self.assertEqual(len(prs_res.prs_score), 20)
        self.assertEqual(len(prs_res.rank), 20)
        self.assertEqual(set(prs_res.rank), set(range(20)))

    def test_refine_sample(self):
        # Misclassified sample test
        x_sample = self.X[70]
        y_true = 1 - self.model.predict(x_sample[None, :])[0]

        X_val = self.X[80:100]
        y_val = self.y[80:100]

        res = refine_sample(self.model, x_sample, y_true, X_accept=X_val, y_accept=y_val)
        self.assertIsInstance(res, RefinementResult)
        self.assertTrue(res.attempted)

    def test_run_active_refinement(self):
        X_pool = self.X[60:90]
        y_pool = self.y[60:90]

        X_accept = self.X[90:105]
        y_accept = self.y[90:105]

        X_test = self.X[105:120]
        y_test = self.y[105:120]

        for strat in ("random", "uncertainty", "pd", "pd_u", "prs"):
            # Clone model for each strategy test
            import copy
            m = copy.deepcopy(self.model)
            res = run_active_refinement(
                m,
                X_pool=X_pool,
                y_pool=y_pool,
                X_accept=X_accept,
                y_accept=y_accept,
                X_test=X_test,
                y_test=y_test,
                strategy=strat,
                budget=5,
                seed=42,
            )
            self.assertIsInstance(res, ActiveLearningResult)
            self.assertEqual(res.labels_queried, 5)
            self.assertEqual(len(res.history), 5)


if __name__ == "__main__":
    unittest.main()
