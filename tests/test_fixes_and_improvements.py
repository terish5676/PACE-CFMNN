"""Unit tests for fixes and improvements in PACE-CFMNN v0.2.1.

Verifies:
1. Tolerant Min-Max scaling and auto-clipping.
2. Safe FMM contraction rollback under the Acceptance Gate.
3. predict_with_trust operational risk scoring.
4. EnsembleCFMNN permutation ensemble for order invariance.
5. detect_label_conflicts and clean_label_conflicts for data pathologies.
6. High-level model.refine guarded by Acceptance Gate.
"""

import unittest
import numpy as np

from pace_cfmnn import (
    CFMNN,
    EnsembleCFMNN,
    safe_apply_fmm_contraction,
    compute_error_risk_score,
    detect_label_conflicts,
    clean_label_conflicts,
)


class TestFixesAndImprovements(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.uniform(0.0, 1.0, size=(100, 4))
        self.y = (self.X[:, 0] + self.X[:, 1] > 1.0).astype(int)

    def test_tolerant_min_max_scaling(self):
        """Float precision variations within [-1e-5, 1+1e-5] should not crash."""
        X_imprecise = self.X.copy()
        X_imprecise[0, 0] = -1e-15
        X_imprecise[1, 1] = 1.0000000000000002

        m = CFMNN(theta=0.3, gamma=4.0)
        # Should fit without raising ValueError
        m.fit(X_imprecise, self.y)
        self.assertTrue(m.n_boxes_ > 0)

        # But severely unscaled data should still raise ValueError
        X_bad = self.X.copy()
        X_bad[0, 0] = 5.0
        with self.assertRaises(ValueError):
            m.fit(X_bad, self.y)

    def test_predict_with_trust(self):
        """predict_with_trust returns predictions, valid risk scores, and trust booleans."""
        m = CFMNN(theta=0.3, gamma=4.0).fit(self.X, self.y)
        preds, risk, is_trusted = m.predict_with_trust(self.X[:20], risk_threshold=0.6)

        self.assertEqual(len(preds), 20)
        self.assertEqual(len(risk), 20)
        self.assertEqual(len(is_trusted), 20)
        self.assertTrue((risk >= 0.0).all() and (risk <= 1.0).all())
        self.assertTrue(is_trusted.dtype == bool)
        np.testing.assert_array_equal(is_trusted, risk < 0.6)

    def test_compute_error_risk_score(self):
        """Direct computation of operational error risk score."""
        m = CFMNN(theta=0.3, gamma=4.0).fit(self.X, self.y)
        r = compute_error_risk_score(m, self.X[:15], alpha=0.5)
        self.assertEqual(len(r), 15)
        self.assertTrue((r >= 0.0).all() and (r <= 1.0).all())

    def test_safe_apply_fmm_contraction_reverts_on_failure(self):
        """Safe contraction should preserve model if contraction hurts validation metrics."""
        m = CFMNN(theta=0.3, gamma=4.0).fit(self.X, self.y)
        V_before = m.V_.copy()
        W_before = m.W_.copy()

        # Call with strict gate parameters
        m_after, passed, reasons = safe_apply_fmm_contraction(
            m, self.X, self.y, margin_macro_f1=10.0, max_collateral_pct=0.0
        )
        # Even if rejected, the model remains functional and equal to before if rolled back
        self.assertEqual(m_after.n_boxes_, len(V_before))
        if not passed:
            np.testing.assert_array_equal(m_after.V_, V_before)
            np.testing.assert_array_equal(m_after.W_, W_before)

    def test_ensemble_cfmnn_order_invariance(self):
        """EnsembleCFMNN averages predictions across randomized permutations."""
        ens = EnsembleCFMNN(n_estimators=3, theta=0.3, gamma=4.0, random_state=42)
        ens.fit(self.X, self.y)

        preds = ens.predict(self.X[:10])
        probs = ens.predict_proba(self.X[:10])

        self.assertEqual(len(preds), 10)
        self.assertEqual(probs.shape, (10, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)

    def test_label_conflict_detection_and_cleaning(self):
        """Detect and clean conflicting duplicate rows."""
        X_with_conflict = np.array([
            [0.2, 0.3],
            [0.2, 0.3],  # Duplicate input with different class
            [0.2, 0.3],  # Duplicate input
            [0.8, 0.9],
        ])
        y_with_conflict = np.array([0, 1, 0, 1])

        report = detect_label_conflicts(X_with_conflict, y_with_conflict)
        self.assertTrue(report["has_conflicts"])
        self.assertEqual(report["conflict_rows"], 3)

        # Clean with majority strategy
        X_clean, y_clean, rep = clean_label_conflicts(X_with_conflict, y_with_conflict, strategy="majority")
        self.assertEqual(len(X_clean), 4)
        # Row 0, 1, 2 should all resolve to majority class 0
        np.testing.assert_array_equal(y_clean[:3], [0, 0, 0])

        # Clean with drop strategy
        X_drop, y_drop, _ = clean_label_conflicts(X_with_conflict, y_with_conflict, strategy="drop")
        self.assertEqual(len(X_drop), 1)
        self.assertEqual(y_drop[0], 1)

    def test_autocfmnn_with_mixed_raw_dataframe(self):
        """AutoCFMNN handles raw DataFrames with strings, booleans, missing values, and unscaled numbers."""
        import pandas as pd
        from pace_cfmnn import AutoCFMNN

        df = pd.DataFrame({
            "age": [25.0, np.nan, 45.0, 50.0, 23.0, 60.0],
            "city": ["New York", "Paris", "New York", "London", "Paris", np.nan],
            "is_smoker": [True, False, False, True, False, False],
            "income": [50000.0, 80000.0, 120000.0, 110000.0, 45000.0, 95000.0],
        })
        y = np.array([0, 1, 1, 1, 0, 1])

        clf = AutoCFMNN(theta=0.4, gamma=4.0, random_state=42)
        clf.fit(df, y)

        preds = clf.predict(df)
        self.assertEqual(len(preds), 6)

        preds, risk, is_trusted = clf.predict_with_trust(df)
        self.assertEqual(len(risk), 6)
        self.assertTrue((risk >= 0.0).all() and (risk <= 1.0).all())

        # Test explain produces DataFrame with original feature names
        exp_df = clf.explain(df.iloc[:2], target="decision")
        self.assertIsInstance(exp_df, pd.DataFrame)
        self.assertEqual(list(exp_df.columns), list(df.columns))

    def test_cfmnn_unified_explain_interface(self):
        """CFMNN.explain provides seamless access to decision, box, and contrastive explanations."""
        m = CFMNN(theta=0.3, gamma=4.0).fit(self.X, self.y)

        # Target B: decision (PACE-flip)
        exp_dec = m.explain(self.X[:5], target="decision")
        self.assertEqual(exp_dec.shape, (5, 4))

        # Target A: box (additive decomposition)
        exp_box = m.explain(self.X[:5], target="box")
        self.assertEqual(exp_box.shape, (5, 4))

        # Target C: contrastive
        exp_con = m.explain(self.X[:5], target="contrastive")
        self.assertTrue(hasattr(exp_con, "e"))
        self.assertEqual(exp_con.e.shape, (5, 4))

    def test_exact_cache_candidate_bounding_equivalence(self):
        """ExactStructuralCache with candidate bounding matches full recompute exactly."""
        from pace_cfmnn import ExactStructuralCache
        from pace_cfmnn.explain import decomposition

        m = CFMNN(theta=0.3, gamma=4.0).fit(self.X, self.y)
        cache = ExactStructuralCache(m, background=self.X[:30], rho=0.05)

        rng = np.random.default_rng(123)
        for i in range(5):
            x = self.X[i]
            perts = np.clip(x + rng.uniform(-0.05, 0.05, size=(10, 4)), 0.0, 1.0)
            phi_cached, _ = cache.explain_perturbed_batch(x, perts)
            phi_recompute = decomposition(m, perts, self.X[:30]).phi

            # Attribution error must be 0.0 to machine precision
            np.testing.assert_allclose(phi_cached, phi_recompute, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
