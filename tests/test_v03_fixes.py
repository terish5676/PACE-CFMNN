"""Tests for v0.3.0 enhancements:
1. Oblique / Rotated Hyperbox CFMNN (ObliqueCFMNN, RotatedHyperbox).
2. Pretrained Vision/Text Embedding Hooks (EmbeddedCFMNN).
3. Scalable Active Learning (Pre-clustering, mini-batch subsampling, caching).
4. Elimination of Within-Class Explanation Instability (EI_within) from primary risk score.
"""

from __future__ import annotations

import unittest
import numpy as np

from pace_cfmnn import (
    CFMNN,
    ObliqueCFMNN,
    RotatedHyperbox,
    EmbeddedCFMNN,
    compute_risk,
    compute_prs,
    run_active_refinement,
    generate_perturbations,
    compute_prediction_disagreement,
    compute_uncertainty,
    compute_explanation_instability,
)


class TestObliqueCFMNN(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset with a diagonal decision boundary: x0 + x1 > 1.0
        rng = np.random.default_rng(42)
        n = 100
        X = rng.uniform(0.0, 1.0, size=(n, 2))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        self.X, self.y = X, y

    def test_oblique_cfmnn_fit_predict(self):
        model = ObliqueCFMNN(rotation="pca", max_hyperbox_size=0.4)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        acc = float(np.mean(preds == self.y))
        self.assertGreaterEqual(acc, 0.85)
        self.assertGreater(model.n_boxes_, 0)

    def test_oblique_cfmnn_explain_back_projection(self):
        model = ObliqueCFMNN(rotation="pca", max_hyperbox_size=0.4)
        model.fit(self.X, self.y)
        # Explain first 5 points
        phi_x = model.explain(self.X[:5], target="decision")
        self.assertEqual(phi_x.shape, (5, 2))
        # Attributions should be finite numbers
        self.assertTrue(np.all(np.isfinite(phi_x)))

    def test_rotated_hyperbox_membership(self):
        # 45-degree rotation matrix
        theta = np.pi / 4.0
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        c = np.array([0.5, 0.5])
        v = np.array([-0.2, -0.2])
        w = np.array([0.2, 0.2])
        
        box = RotatedHyperbox(cls=1, v=v, w=w, c=c, R=R, centroid=c)
        # Center should have membership 1.0
        m_center = box.membership(c)
        self.assertAlmostEqual(m_center, 1.0)
        # Point outside on both rotated dimensions should have membership 0.0
        m_far = box.membership(np.array([10.0, 0.0]))
        self.assertAlmostEqual(m_far, 0.0)
        # Point outside on 1 of 2 dimensions has membership 0.5
        m_diag = box.membership(np.array([5.0, 5.0]))
        self.assertAlmostEqual(m_diag, 0.5)


class MockTensor:
    """Mock PyTorch tensor supporting .detach().cpu().numpy()."""
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class TestEmbeddedCFMNN(unittest.TestCase):
    def test_text_encoder_pipeline(self):
        texts = [
            "acute myocardial infarction",
            "coronary heart failure",
            "severe chest pain cardiac",
            "common seasonal cold",
            "mild cough and sneezing",
            "rhinovirus viral infection",
        ]
        labels = np.array([1, 1, 1, 0, 0, 0])

        # Simple mock bag-of-words / frequency encoder
        vocab = ["infarction", "coronary", "chest", "cold", "cough", "rhinovirus"]
        def mock_text_encoder(raw_texts):
            vectors = []
            for t in raw_texts:
                vec = [float(t.count(word)) for word in vocab]
                vectors.append(vec)
            return np.array(vectors)

        model = EmbeddedCFMNN(encoder=mock_text_encoder, max_hyperbox_size=0.5)
        model.fit(texts, labels)
        preds = model.predict(texts)
        self.assertTrue(np.all(preds == labels))

        # Check explainability on embeddings
        phi = model.explain(texts[:2])
        self.assertEqual(phi.shape, (2, len(vocab)))

    def test_torch_tensor_encoder(self):
        rng = np.random.default_rng(123)
        raw_images = rng.uniform(0.0, 1.0, size=(10, 3, 32, 32))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Mock CNN feature extractor returning a Tensor-like object
        def mock_cnn_encoder(images):
            # Compress 3x32x32 into 8 continuous feature embeddings
            flat_features = np.mean(images, axis=(2, 3))  # shape (10, 3)
            extra = np.std(images, axis=(2, 3))          # shape (10, 3)
            combined = np.hstack([flat_features, extra]) # shape (10, 6)
            return MockTensor(combined)

        model = EmbeddedCFMNN(encoder=mock_cnn_encoder, max_hyperbox_size=0.5, use_oblique=True)
        model.fit(raw_images, y)
        preds = model.predict(raw_images)
        self.assertEqual(len(preds), 10)


class TestScalableActiveLearning(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 100
        d = 4
        self.X_pool = rng.uniform(0.1, 0.9, size=(n, d))
        self.y_pool = (self.X_pool[:, 0] > 0.5).astype(int)
        
        # Initial small training set
        X_init = self.X_pool[:10]
        y_init = self.y_pool[:10]
        self.model = CFMNN(max_hyperbox_size=0.3)
        self.model.fit(X_init, y_init)

        self.X_accept = self.X_pool[10:20]
        self.y_accept = self.y_pool[10:20]
        self.X_test = self.X_pool[20:30]
        self.y_test = self.y_pool[20:30]

    def test_cluster_margin_active_learning(self):
        res = run_active_refinement(
            self.model,
            self.X_pool,
            self.y_pool,
            self.X_accept,
            self.y_accept,
            self.X_test,
            self.y_test,
            strategy="cluster_margin",
            budget=5,
            n_clusters=4,
            seed=42,
        )
        self.assertEqual(res.labels_queried, 5)
        self.assertEqual(len(res.history), 5)

    def test_minibatch_subsampling(self):
        res = run_active_refinement(
            self.model,
            self.X_pool,
            self.y_pool,
            self.X_accept,
            self.y_accept,
            self.X_test,
            self.y_test,
            strategy="prs",
            budget=4,
            batch_sample_size=20,  # evaluate on mini-batch of 20
            seed=42,
        )
        self.assertEqual(res.labels_queried, 4)


class TestEIExclusionFromRiskScore(unittest.TestCase):
    def test_compute_risk_formula(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 0.9, size=(30, 4))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        model = CFMNN(max_hyperbox_size=0.4)
        model.fit(X, y)

        # Operational risk is strictly 0.5*PD + 0.5*U
        risk = compute_risk(model, X, alpha=0.5, seed=42)
        self.assertEqual(len(risk), len(X))
        self.assertTrue(np.all((risk >= 0.0) & (risk <= 1.0)))

        # Verify compute_prs with default weights matches compute_risk
        prs_res = compute_prs(model, X, seed=42, weights=(0.5, 0.5, 0.0))
        np.testing.assert_allclose(prs_res.prs_score, risk, atol=1e-6)

        # Verify rank order is identical
        expected_rank = np.empty(len(X), dtype=int)
        expected_rank[np.argsort(-risk)] = np.arange(len(X))
        np.testing.assert_array_equal(prs_res.rank, expected_rank)


if __name__ == "__main__":
    unittest.main()
