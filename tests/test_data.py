"""Tests for datasets, splits and scaling."""

import unittest

import numpy as np

from pace_cfmnn import data


class TestSynthetic(unittest.TestCase):
    def test_reproducible_and_shaped(self):
        a = data.make_synthetic("S1", n=1000, seed=3)
        b = data.make_synthetic("S1", n=1000, seed=3)
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y, b.y)
        self.assertEqual(a.X.shape, (1000, 10))
        self.assertTrue(0.4 < a.y.mean() < 0.6)

    def test_s1_label_depends_only_on_features_0_to_2(self):
        ds = data.make_synthetic("S1", n=2000, seed=1)
        rule = (data.synth_score(ds.X) > data.synth_threshold()).astype(int)
        np.testing.assert_array_equal(ds.y, rule)
        # S1 keeps a margin around the boundary, so it is separable.
        self.assertTrue((np.abs(data.synth_score(ds.X) - 1.25) > 0.1).all())

    def test_s2_overlaps(self):
        ds = data.make_synthetic("S2", n=4000, seed=1)
        rule = (data.synth_score(ds.X) > data.synth_threshold()).astype(int)
        disagree = (ds.y != rule).mean()
        self.assertTrue(0.05 < disagree < 0.3)

    def test_variants(self):
        imb = data.make_synthetic("S2", n=5000, seed=2, minority=0.1)
        self.assertTrue(0.05 < imb.y.mean() < 0.15)
        cor = data.make_synthetic("S1", n=2000, seed=2, correlated=True)
        self.assertGreater(np.corrcoef(cor.X[:, 0], cor.X[:, 3])[0, 1], 0.9)
        shifted = data.make_synthetic("S1", n=2000, seed=2, shift=1.0)
        self.assertGreater(shifted.X[:, 0].mean(), 0.6)


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.y = np.random.default_rng(0).choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=3000)

    def _check_partition(self, parts):
        allidx = np.concatenate(list(parts.values()))
        self.assertEqual(len(allidx), len(self.y))
        self.assertEqual(len(np.unique(allidx)), len(self.y))

    def test_role_split_shares_and_stratification(self):
        parts = data.role_split(self.y, seed=4)
        self._check_partition(parts)
        for role, share in data.ROLE_SHARES.items():
            self.assertAlmostEqual(len(parts[role]) / len(self.y), share, delta=0.01)
            for c in (0, 1, 2):
                self.assertAlmostEqual((self.y[parts[role]] == c).mean(),
                                       (self.y == c).mean(), delta=0.03)
        again = data.role_split(self.y, seed=4)
        for role in parts:
            np.testing.assert_array_equal(parts[role], again[role])

    def test_paper_split(self):
        parts = data.paper_split(self.y, seed=1)
        self._check_partition(parts)
        self.assertAlmostEqual(len(parts["test"]) / len(self.y), 0.2, delta=0.01)

    def test_scaler_uses_training_rows_and_clips(self):
        X = np.random.default_rng(1).normal(size=(500, 4))
        sc = data.fit_scaler(X[:250])
        tr, te = sc.transform(X[:250]), sc.transform(X[250:])
        self.assertAlmostEqual(tr.min(), 0.0)
        self.assertAlmostEqual(tr.max(), 1.0)
        self.assertTrue((te >= 0).all() and (te <= 1).all())


class TestBundled(unittest.TestCase):
    def test_bundled_sets_load(self):
        for name, shape in [("wine", (178, 13)), ("breast_cancer", (569, 30)),
                            ("digits", (1797, 64))]:
            ds = data.load(name)
            self.assertEqual(ds.X.shape, shape)

    def test_uci_sets_match_paper_table_1(self):
        expected = {"thyroid": (7200, 21, 3), "shuttle": (58000, 9, 7),
                    "skin": (245057, 3, 2)}
        for name, (rows, feats, classes) in expected.items():
            with self.subTest(name=name):
                try:
                    ds = data.load(name)
                except FileNotFoundError:
                    self.skipTest(f"{name} not downloaded")
                self.assertEqual(ds.X.shape, (rows, feats))
                self.assertEqual(len(np.unique(ds.y)), classes)

    def test_missing_uci_file_message(self):
        with self.assertRaises(FileNotFoundError) as cm:
            data.load("thyroid", data_dir=data.DATA_DIR / "__missing__")
        self.assertIn("ann-train.data", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
