"""Tests for the front end (pace_cfmnn.api): it must call the paper's classes unchanged."""
import copy
import unittest

import numpy as np
from sklearn.base import clone

from pace_cfmnn import CFMNN, PaceClassifier
from pace_cfmnn import api
from pace_cfmnn.evaluate import classification_metrics
from pace_cfmnn.explain import PaceFlip, PaceShap


def noisy_problem(n=300, seed=0):
    """Two overlapping classes in 3 features, so CFMNN makes tied training errors."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    y = (X[:, 0] + X[:, 1] + 0.3 * rng.standard_normal(n) > 1.0).astype(int)
    return X, y


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = noisy_problem()
        cls.model = CFMNN(theta=0.3).fit(cls.X, cls.y)
        cls.bg = cls.X[:50]
        cls.pts = cls.X[200:220]

    def test_explain_shap_is_pace_shap_and_adds_up(self):
        phi = api.explain(self.model, self.pts, self.bg)
        ref, fx, f0 = PaceShap(self.model, self.bg).explain(self.pts, return_parts=True)
        self.assertEqual(phi.shape, (len(self.pts), 3))
        np.testing.assert_array_equal(phi, ref)
        np.testing.assert_allclose(phi.sum(axis=1), fx - f0, atol=1e-12)

    def test_explain_flip_is_pace_flip(self):
        score = api.explain(self.model, self.pts, self.bg, method="flip")
        np.testing.assert_array_equal(score, PaceFlip(self.model, self.bg).explain(self.pts))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            api.explain(self.model, self.pts, self.bg, method="lime")

    def test_repair_leaves_the_input_model_alone_and_passes_the_gate(self):
        before_pred = self.model.predict(self.X)
        before_boxes = self.model.V_.copy(), self.model.W_.copy()
        result = api.repair(self.model, self.X, self.y, budget=40, n_rounds=2)
        np.testing.assert_array_equal(self.model.predict(self.X), before_pred)
        np.testing.assert_array_equal(self.model.V_, before_boxes[0])
        np.testing.assert_array_equal(self.model.W_, before_boxes[1])
        self.assertIsInstance(result.model, CFMNN)
        self.assertIsNone(result.certificate)
        self.assertGreater((before_pred != self.y).sum(), 0, "the toy problem should have errors")
        self.assertGreater(len(result.edits), 0, "the toy problem should give an accepted edit")
        f1_before = classification_metrics(self.y, before_pred)["macro_f1"]
        f1_after = classification_metrics(self.y, result.model.predict(self.X))["macro_f1"]
        # The gate compares each edit with the model before it, so drops can add up over edits.
        self.assertGreater(f1_after, f1_before - 0.5 * len(result.edits) - 1e-9)

    def test_repair_certificate_holds_every_changed_prediction(self):
        grid = np.random.default_rng(1).random((2000, 3))
        tight = api.repair(self.model, self.X, self.y, certify_on=grid, budget=40, n_rounds=2)
        loose = api.repair(self.model, self.X, self.y, certify_on=grid, budget=40, n_rounds=2,
                           tight=False)
        np.testing.assert_array_equal(tight.model.predict(grid), loose.model.predict(grid))
        changed = self.model.predict(grid) != tight.model.predict(grid)
        self.assertTrue(changed.any(), "the repair should change some predictions on the grid")
        self.assertFalse((changed & ~tight.certificate).any())
        self.assertFalse((tight.certificate & ~loose.certificate).any())

    def test_certify_one_edit_holds_every_changed_prediction(self):
        grid = np.random.default_rng(4).random((2000, 3))
        before = self.model.predict(grid)
        hit_any = False
        for k in range(0, self.model.n_boxes_, max(1, self.model.n_boxes_ // 10)):
            edited = copy.deepcopy(self.model)
            edited._remove_box(k)
            changed = before != edited.predict(grid)
            cert = api.certify(self.model, edited, grid, removed=k)
            self.assertFalse((changed & ~cert).any(), f"box {k}")
            hit_any |= changed.any()
        self.assertTrue(hit_any, "some removal should change a prediction")


class TestPaceClassifier(unittest.TestCase):
    def test_fit_keeps_background_and_methods_match_the_functions(self):
        X, y = noisy_problem(seed=2)
        model = PaceClassifier(theta=0.3).fit(X, y)
        self.assertEqual(model.background_.shape, (100, 3))
        self.assertTrue(all(any((row == X).all(axis=1)) for row in model.background_))
        np.testing.assert_array_equal(model.explain(X[:5]),
                                      api.explain(model, X[:5], model.background_))
        result = model.repair(X, y, certify_on=X[:50], budget=20, n_rounds=1)
        ref = api.repair(model, X, y, certify_on=X[:50], budget=20, n_rounds=1)
        np.testing.assert_array_equal(result.certificate, ref.certificate)
        np.testing.assert_array_equal(result.model.predict(X), ref.model.predict(X))

    def test_works_with_sklearn_clone_and_matches_cfmnn(self):
        X, y = noisy_problem(seed=3)
        model = clone(PaceClassifier(theta=0.2))
        self.assertEqual(model.get_params()["theta"], 0.2)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), CFMNN(theta=0.2).fit(X, y).predict(X))


if __name__ == "__main__":
    unittest.main()
