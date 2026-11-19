"""Run B: the brute-force counterfactual, the DiCE method keyword, the robustness measures, the
summary's directions and one small end-to-end run."""

import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_cf_robustness as rb  # noqa: E402
from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.counterfactual import brute_force_counterfactual, replacement_values  # noqa: E402
from pace_cfmnn.neighborhood import Perturber  # noqa: E402


def toy(seed=0, n=300, d=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, n)) > 0.75).astype(int)
    return X, y


def naive_best(model, x, target, values, locked, max_features=2):
    """Every change of 1 or 2 mutable features to a replacement value, one by one."""
    mut = [i for i in range(len(x)) if not locked[i]]
    best = (np.inf, None)
    for k in range(1, max_features + 1):
        for feats in itertools.combinations(mut, k):
            for vals in itertools.product(*[values[i] for i in feats]):
                c = x.copy()
                c[list(feats)] = vals
                if model.predict(c[None])[0] == target:
                    l1 = float(np.abs(c - x).sum())
                    if l1 < best[0]:
                        best = (l1, c)
    return best


class TestBruteForce(unittest.TestCase):
    def setUp(self):
        self.X, self.y = toy()
        self.model = CFMNN().fit(self.X, self.y)
        self.values = replacement_values(self.X, 6)

    def test_replacement_values_occur_in_the_training_data(self):
        for i, v in enumerate(self.values):
            self.assertLessEqual(len(v), 6)
            self.assertTrue(np.all(np.isin(v, self.X[:, i])))
            self.assertTrue(np.all(np.diff(v) > 0))                 # sorted, no repeats

    def test_equals_the_naive_search_and_is_valid(self):
        rng = np.random.default_rng(1)
        found = 0
        for r in rng.choice(len(self.X), 12, replace=False):
            x = self.X[r]
            target = 1 - self.model.predict(x[None])[0]
            cf = brute_force_counterfactual(self.model, x, target, self.values)
            l1, best = naive_best(self.model, x, target, self.values, np.zeros(4, bool))
            self.assertEqual(cf.found, best is not None)
            if cf.found:
                found += 1
                self.assertAlmostEqual(cf.l1, l1, places=12)           # the same minimum
                self.assertEqual(self.model.predict(cf.x_cf[None])[0], target)
                self.assertLessEqual(cf.n_changed, 2)
                self.assertEqual(cf.n_changed, int(np.sum(np.abs(cf.x_cf - x) > 1e-9)))
        self.assertGreater(found, 5)

    def test_locked_features_never_change_and_one_feature_only(self):
        x = self.X[3]
        target = 1 - self.model.predict(x[None])[0]
        lock = np.array([True, True, False, False])
        cf = brute_force_counterfactual(self.model, x, target, self.values, locked=lock)
        l1, best = naive_best(self.model, x, target, self.values, lock)
        self.assertEqual(cf.found, best is not None)
        if cf.found:
            self.assertTrue(np.array_equal(cf.x_cf[lock], x[lock]))
            self.assertAlmostEqual(cf.l1, l1, places=12)
        one = brute_force_counterfactual(self.model, x, target, self.values, max_features=1)
        if one.found:
            self.assertEqual(one.n_changed, 1)

    def test_not_found_when_nothing_in_reach(self):
        x = self.X[0]
        target = 1 - self.model.predict(x[None])[0]
        none = brute_force_counterfactual(self.model, x, target, [np.array([x[i]]) for i in range(4)])
        self.assertFalse(none.found)
        self.assertIsNone(none.x_cf)


class TestDiceMethod(unittest.TestCase):
    def test_default_is_random_and_unchanged(self):
        X, y = toy(2, n=200)
        m = CFMNN().fit(X, y)
        x = X[5]
        target = 1 - m.predict(x[None])[0]
        a = xb.dice_counterfactual(m, x, X, y, target, seed=3, n_cfs=2, sparse=True, pick="nearest_valid")
        b = xb.dice_counterfactual(m, x, X, y, target, seed=3, n_cfs=2, sparse=True, pick="nearest_valid",
                                   method="random")
        self.assertIsNotNone(a)
        self.assertTrue(np.array_equal(a, b))

    def test_other_methods_return_an_answer_or_none(self):
        X, y = toy(2, n=200)
        m = CFMNN().fit(X, y)
        x = X[5]
        target = 1 - m.predict(x[None])[0]
        for method in ("genetic", "kdtree"):
            z = xb.dice_counterfactual(m, x, X, y, target, seed=1, n_cfs=1, sparse=False, method=method)
            self.assertTrue(z is None or np.asarray(z).shape == (4,))
        with self.assertRaises(ValueError):
            xb.dice_counterfactual(m, x, X, y, target, method="nope")


class TestMeasures(unittest.TestCase):
    def test_input_sensitivity(self):
        x, cf = np.zeros(3), np.array([1.0, 0.0, 0.0])
        copies = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0], [0.0, 0, 0.4]])     # one copy is x itself
        # the answer moves by 0.2 (copy 1), 0.4 (copy 2), 0.8 (copy 4); the copies move by .1, .2, .4
        regen = {0.1: [1.2, 0, 0], 0.2: [1.0, 0.4, 0], 0.4: [1.0, 0, 0.8]}

        def regenerate(z):
            return np.array(regen[round(float(np.abs(z).sum()), 1)], float)

        r1, n = rb.input_sensitivity(x, cf, copies, regenerate)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(r1, 2.0)           # ratios 2, 2, 2
        r1, n = rb.input_sensitivity(x, cf, copies, lambda z: None)
        self.assertTrue(np.isnan(r1) and n == 0)
        # a missing counterfactual for one copy leaves it out
        r1, n = rb.input_sensitivity(x, cf, copies, lambda z: None if z[1] > 0 else regenerate(z))
        self.assertEqual(n, 2)

    def test_answer_noise_and_refit_validity(self):
        X, y = toy(4)
        m = CFMNN().fit(X, y)
        pert = Perturber(kappa=0.05, k=20).fit(X)
        deep = X[np.argmax(m.predict_details(X).support_rescaled[:, 1])]
        r2 = rb.answer_noise(m, deep, 1, pert, seed=1)
        self.assertEqual(r2, rb.answer_noise(m, deep, 1, pert, seed=1))             # seeded
        self.assertTrue(0.0 <= r2 <= 1.0)
        self.assertGreater(r2, 0.5)
        far = X[np.argmin(m.predict_details(X).support_rescaled[:, 1])]
        self.assertLess(rb.answer_noise(m, far, 1, pert, seed=1), r2)
        others = [CFMNN().fit(X[p], y[p]) for p in (np.random.default_rng(j).permutation(len(y)) for j in range(4))]
        self.assertEqual(rb.refit_validity(others, deep, 1), float(np.mean([o.predict(deep[None])[0] == 1 for o in others])))
        self.assertIn(rb.refit_validity(others, deep, 1), {0.0, 0.25, 0.5, 0.75, 1.0})


class TestSummary(unittest.TestCase):
    def table(self, pace_r1, dice_r1, pace_r3, dice_r3):
        rows = []
        for s in range(110, 120):
            for method, r1, r3 in (("PACE", pace_r1, pace_r3), ("DiCE random", dice_r1, dice_r3),
                                   ("PACE-sparse", 1.0, 0.5)):
                for point in range(3):
                    row = dict.fromkeys(rb.COLUMNS, np.nan)
                    row.update(dataset="wine", seed=s, point=point, method=method, found=True,
                               valid=True, ms=1.0, l1=1.0, n_changed=1, plausibility=1.0, r2=1.0, r4=0.1,
                               r1=r1 + 0.01 * (s - 110) + 0.001 * point, r3=r3 + 0.01 * (s - 110) % 0.1)
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_directions(self):
        # r1: lower is better; r3: higher is better
        df = self.table(pace_r1=0.5, dice_r1=1.0, pace_r3=0.9, dice_r3=0.4)
        self.assertEqual(rb.compare(df, "PACE", "DiCE random", "r1", True).verdict.tolist(), ["better"])
        self.assertEqual(rb.compare(df, "PACE", "DiCE random", "r3", False).verdict.tolist(), ["better"])
        df = self.table(pace_r1=1.5, dice_r1=1.0, pace_r3=0.2, dice_r3=0.7)
        self.assertEqual(rb.compare(df, "PACE", "DiCE random", "r1", True).verdict.tolist(), ["worse"])
        self.assertEqual(rb.compare(df, "PACE", "DiCE random", "r3", False).verdict.tolist(), ["worse"])
        df = self.table(1.0, 1.0, 0.5, 0.5)
        self.assertEqual(rb.compare(df, "PACE", "DiCE random", "r1", True).verdict.tolist(), ["level"])

    def test_summary_text(self):
        df = self.table(0.5, 1.0, 0.9, 0.4)
        with tempfile.TemporaryDirectory() as tmp:
            text = rb.summarise(df, Path(tmp))
            self.assertTrue((Path(tmp) / "summary.md").exists())
        self.assertIn("**PACE vs DiCE random, r1** (lower is better): better 1, level 0, worse 0", text)
        self.assertIn("**PACE-sparse vs DiCE random, r3** (higher is better)", text)


class TestEndToEnd(unittest.TestCase):
    def test_a_small_dataset_has_every_method_and_column(self):
        # DiCE takes seconds per call, so it is replaced here by a stub that records its method;
        # the real DiCE calls are covered by TestDiceMethod.
        seen = []

        def stub(model, x, X_train, y_train, target, seed=0, locked=(), method="random", **kw):
            seen.append(method)
            pts = X_train[model.predict(X_train) == target]
            return pts[np.abs(pts - x).sum(axis=1).argmin()] if len(pts) else None

        real = rb.xb.dice_counterfactual
        rb.xb.dice_counterfactual = stub
        try:
            rows = rb.run_one("S1", 120, n=2)
        finally:
            rb.xb.dice_counterfactual = real
        self.assertEqual(set(seen), {"random", "genetic", "kdtree"})
        self.assertTrue(rows)
        df = pd.DataFrame(rows)
        self.assertEqual(list(df.columns), rb.COLUMNS)
        self.assertEqual(set(df.method), set(rb.METHODS))
        for _, r in df.iterrows():
            if r.found:
                self.assertTrue(0 <= r.r2 <= 1 and 0 <= r.r3 <= 1)
            if r.method in ("DiCE genetic", "DiCE kdtree"):
                self.assertTrue(np.isnan(r.r1))                     # left out of r1 by design
        self.assertEqual(set(rb.R1_METHODS), set(rb.METHODS) - {"DiCE genetic", "DiCE kdtree"})
        bf = df[df.method == "Brute force"]
        self.assertTrue(((bf.n_changed <= 2) | bf.n_changed.isna()).all())


if __name__ == "__main__":
    unittest.main()
