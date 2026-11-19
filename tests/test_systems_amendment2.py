"""Run G, Amendment 2: TreeSHAP follows the tree's own paths when a float64 threshold
rounds up to a float32 input value."""

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_systems as rs  # noqa: E402


def rounding_up_case():
    """float32 values a < v < c, where the tree's threshold between a and c (their float64
    midpoint) rounds up to v in float32. scikit-learn sends v right of that threshold."""
    a = np.float32(0.3)
    for k in range(5, 200, 2):                     # k float32 steps apart, more than 1e-7
        c = a
        for _ in range(k):
            c = np.nextafter(c, np.float32(np.inf))
        mid = np.float64(a) / 2 + np.float64(c) / 2
        v = np.float32(mid)
        if np.float64(v) > mid:
            return a, v, c
    raise AssertionError("no case found")


class TestFloat32Floor(unittest.TestCase):
    def test_largest_float32_at_or_below(self):
        rng = np.random.default_rng(0)
        t = np.concatenate([rng.uniform(-1, 1, 5000), [-2.0, 0.0, 0.5]])
        f = rs.float32_floor(t)
        self.assertTrue(np.all(f <= t))
        self.assertTrue(np.array_equal(f, f.astype(np.float32).astype(np.float64)))   # float32 values
        nxt = np.nextafter(f.astype(np.float32), np.float32(np.inf)).astype(np.float64)
        self.assertTrue(np.all(nxt > t))
        # for every float32 x, x <= t exactly when x <= floor(t)
        x = np.concatenate([f, nxt]).astype(np.float32).astype(np.float64)
        T, F = np.tile(t, 2), np.tile(f, 2)
        self.assertTrue(np.array_equal(x <= T, x <= F))


class TestTreeShapPaths(unittest.TestCase):
    def test_point_on_a_rounded_threshold(self):
        warnings.filterwarnings("ignore")
        a, v, c = rounding_up_case()
        X = np.array([[a], [c]], np.float64)
        y = np.array([0, 1])
        tree = DecisionTreeClassifier(random_state=0).fit(X, y)
        th = tree.tree_.threshold[0]
        self.assertTrue(np.float64(v) > th and np.float32(th) == v)   # the case that failed
        x = np.array([[v]], np.float64)
        self.assertEqual(tree.predict(x)[0], 1)              # the tree sends v right
        sv = rs.tree_shap_values(tree, x, X[:1])             # raises if additivity fails
        p, p0 = tree.predict_proba(x)[0], tree.predict_proba(X[:1])[0]
        self.assertTrue(np.allclose(sv.sum(axis=1)[0], p - p0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
