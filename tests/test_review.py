"""Tests for the review's new code (results/review_v1): the exact baseline, raw
margin AUCs, the explainer rows of the stability test, the generic ensembles and
save-as-you-go running."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import trust_combine as tc  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import append_rows, finished, pending  # noqa: E402


def small(seed=0, n=150, d=4, three_classes=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    s = X[:, 0] + X[:, 1]
    y = np.digitize(s, [0.7, 1.3]) if three_classes else (s > 1).astype(int)
    return CFMNN().fit(X, y), X, y


class TestExactShap(unittest.TestCase):
    def test_equals_pace_shap_on_the_same_rows(self):
        m, X, _ = small(three_classes=True)
        np.testing.assert_allclose(xb.exact_shap(m, X[:6], X[100:130]),
                                   PaceShap(m, X[100:130]).explain(X[:6]), atol=1e-9)


class TestRawMarginAUC(unittest.TestCase):
    def setUp(self):
        import run_faithfulness as rf
        self.rf = rf
        self.m, X, _ = small(1, three_classes=True)
        self.X, self.bg = X[:7], X[90:120]
        self.attr = np.random.default_rng(2).normal(size=self.X.shape)

    def by_hand(self, insertion, rng_seed):
        """The raw margin AUC from its definition, with the tests' own random rows."""
        rf, m, X, bg = self.rf, self.m, self.X, self.bg
        n, d = X.shape
        cls = m.predict_details(X).cls
        order = rf._order(self.attr, signed=True)
        rng = np.random.default_rng(rng_seed)
        aucs = []
        for _ in range(10):
            B = bg[rng.integers(len(bg), size=n)]
            Z, src = (B.copy(), X) if insertion else (X.copy(), B)
            curve = [rf._margin(m.predict_details(Z), cls)]
            for k in range(d):
                idx = order[:, k]
                Z[np.arange(n), idx] = src[np.arange(n), idx]
                curve.append(rf._margin(m.predict_details(Z), cls))
            aucs.append(np.trapezoid(np.array(curve), dx=1.0 / d, axis=0))
        return np.mean(aucs, axis=0)

    def test_deletion(self):
        old = self.rf.deletion_auc(self.m, self.X, self.attr, self.bg, seed=3, signed=True)
        new = self.rf.deletion_auc(self.m, self.X, self.attr, self.bg, seed=3, signed=True,
                                   with_raw=True)
        np.testing.assert_array_equal(old[0], new[0])           # earlier outputs unchanged
        np.testing.assert_array_equal(old[1], new[1])
        np.testing.assert_allclose(new[2], self.by_hand(False, 3))

    def test_insertion(self):
        old = self.rf.insertion_auc(self.m, self.X, self.attr, self.bg, seed=2)
        new = self.rf.insertion_auc(self.m, self.X, self.attr, self.bg, seed=2, with_raw=True)
        np.testing.assert_array_equal(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])
        np.testing.assert_allclose(new[2], self.by_hand(True, 3))   # insertion uses seed + 1


class TestExplainerRows(unittest.TestCase):
    def test_fixed_rows_are_v3s_and_new_rows_are_unseen(self):
        import run_faithfulness as rf
        import run_review_rq1 as rr
        rng = np.random.default_rng(0)
        rows = lambda A: {tuple(r) for r in A}               # noqa: E731
        for n_train in (1000, 300):                          # enough unseen rows; too few
            Xtr = rng.uniform(size=(n_train, 3))
            ev, bex = rf.backgrounds_v3(Xtr, 7)
            fixed, new = rr.explainer_rows(Xtr, 7, "fixed"), rr.explainer_rows(Xtr, 7, "new")
            np.testing.assert_array_equal(fixed, bex)
            self.assertEqual(new.shape, bex.shape)
            if n_train >= 3 * len(bex):
                self.assertFalse(rows(new) & (rows(ev) | rows(bex)))
            else:                                            # bootstrap of the explainer rows
                self.assertTrue(rows(new) <= rows(bex))


class TestGenericEnsembles(unittest.TestCase):
    def test_features_are_shares_of_five_members(self):
        m, X, y = small(3, three_classes=True)
        bag = tc.bagged_members(m, X, y, n_models=5, seed=1)
        sgd = tc.sgd_order_members(X, y, n_models=5, seed=1)
        F = tc.ensemble_features(X[:40], m.predict(X[:40]), bag, sgd)
        self.assertEqual(set(F), set(tc.ENSEMBLES))
        for v in F.values():
            self.assertEqual(v.shape, (40,))
            np.testing.assert_allclose(v * 5, np.round(v * 5))
            self.assertTrue(v.min() >= 0 and v.max() <= 1)

    def test_bagged_members_keep_the_row_order(self):
        from sklearn.base import BaseEstimator

        class Recorder(BaseEstimator):
            def fit(self, X, y):
                self.rows_ = X[:, 0].copy()
                return self

        X = np.arange(60, dtype=float)[:, None]               # each row holds its own index
        for m in tc.bagged_members(Recorder(), X, np.zeros(60), n_models=3, seed=0):
            self.assertTrue(np.all(np.diff(m.rows_) >= 0))     # in order, with repeats
            self.assertLess(len(np.unique(m.rows_)), 60)       # a bootstrap sample

    def test_sgd_members_differ_only_by_order(self):
        _, X, y = small(4)
        a = tc.sgd_order_members(X, y, n_models=2, seed=0)
        b = tc.sgd_order_members(X, y, n_models=2, seed=0)
        np.testing.assert_array_equal(a[0].coef_, b[0].coef_)      # seeded
        self.assertFalse(np.array_equal(a[0].coef_, a[1].coef_))   # different orders


class TestResume(unittest.TestCase):
    def test_append_finished_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "runs.csv"
            tasks = [("S1", 0, 5), ("S1", 1, 5), ("wine", 0, 5)]
            self.assertEqual(pending(p, tasks), tasks)
            append_rows(p, [{"dataset": "S1", "seed": 0, "method": "a", "x": 1.0}])
            append_rows(p, [{"x": 2.0, "method": "b", "dataset": "wine", "seed": 0}])  # other order
            self.assertEqual(pd.read_csv(p).x.tolist(), [1.0, 2.0])
            self.assertEqual(finished(p), {("S1", 0), ("wine", 0)})
            with self.assertRaises(SystemExit):              # results exist: --resume or --fresh
                pending(p, tasks)
            self.assertEqual(pending(p, tasks, resume=True), [("S1", 1, 5)])
            self.assertEqual(pending(p, tasks, fresh=True), tasks)
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()
