"""Unit tests for Simpson (1992) traditional FMM contraction and refinement module."""

import unittest
import numpy as np

from pace_cfmnn import CFMNN
from pace_cfmnn.contraction import apply_fmm_contraction, contract_boxes, test_box_overlap
from pace_cfmnn.refinement import (
    apply_explanation_guided_shrink,
    apply_random_shrink_control,
    check_acceptance_gate,
    run_nested_cross_fitted_refinement,
)


class TestContraction(unittest.TestCase):
    def test_overlap_detection_and_contraction_case1(self):
        # Case 1: v_j < v_k < w_j < w_k
        v_j = np.array([0.1, 0.2])
        w_j = np.array([0.5, 0.6])
        v_k = np.array([0.4, 0.5])
        w_k = np.array([0.8, 0.9])

        overlaps, dim, case = test_box_overlap(v_j, w_j, v_k, w_k)
        self.assertTrue(overlaps)
        self.assertEqual(case, 1)

        contract_boxes(v_j, w_j, v_k, w_k, dim, case)
        # After contraction on dim, w_j[dim] == v_k[dim]
        self.assertAlmostEqual(w_j[dim], v_k[dim])

        # Overlap test should now return False
        ov2, _, _ = test_box_overlap(v_j, w_j, v_k, w_k)
        self.assertFalse(ov2)

    def test_apply_fmm_contraction_on_model(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(200, 3))
        y = (X[:, 0] > 0.5).astype(int)
        model = CFMNN(theta=0.3, gamma=4.0).fit(X, y)
        n_before = model.n_boxes_

        m_contracted = apply_fmm_contraction(model)
        self.assertEqual(m_contracted.n_boxes_, n_before)
        # Predict should run without issues
        preds = m_contracted.predict(X[:10])
        self.assertEqual(len(preds), 10)


class TestRefinement(unittest.TestCase):
    def test_acceptance_gate_logic(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(100, 3))
        y = (X[:, 0] > 0.5).astype(int)
        model = CFMNN().fit(X, y)

        # Candidate identical to original should pass with 0 gain
        gate = check_acceptance_gate(model, model, X, y)
        self.assertTrue(gate.passed)
        self.assertEqual(gate.primary_gain, 0.0)
        self.assertEqual(gate.newly_misclassified_count, 0)

    def test_explanation_guided_shrink_runs(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(150, 4))
        # noisy boundary
        y = ((X[:, 0] + X[:, 1] + rng.normal(0, 0.2, 150)) > 1.0).astype(int)
        model = CFMNN().fit(X, y)

        m_shrunk = apply_explanation_guided_shrink(model, X, y, background=X[:50])
        p = m_shrunk.predict(X)
        self.assertEqual(len(p), len(X))

    def test_nested_refinement_pipeline(self):
        rng = np.random.default_rng(10)
        X = rng.uniform(size=(120, 3))
        y = (X[:, 0] > 0.5).astype(int)
        eval_res = run_nested_cross_fitted_refinement(X[:80], y[:80], X[80:], y[80:], n_inner_splits=2)
        self.assertIn("accuracy", eval_res.unrefined_metrics)
        self.assertIn("accuracy", eval_res.explanation_shrink_metrics)
        self.assertIsInstance(eval_res.accepted_method, str)


if __name__ == "__main__":
    unittest.main()
