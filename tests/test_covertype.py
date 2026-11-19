"""Run K: the Covertype loader and the script, on small made-up data (no download needed)."""

import gzip
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import download_data as dd  # noqa: E402
import run_covertype as rk  # noqa: E402
from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.ties import training_error_ties  # noqa: E402


def fake_covertype(n=1500, seed=0) -> data.Dataset:
    """Rows shaped like Covertype: 10 measured features, 44 binary ones, labels 1-7."""
    rng = np.random.default_rng(seed)
    area, soil = rng.integers(0, 4, n), rng.integers(0, 40, n)
    X = np.hstack([rng.integers(0, 4000, size=(n, 10)), np.eye(4)[area], np.eye(40)[soil]]).astype(float)
    score = X[:, 0] / 4000 + 0.3 * area + rng.normal(0, 0.1, n)
    y = 1 + np.digitize(score, np.quantile(score, np.arange(1, 7) / 7))
    return data.Dataset("covertype", X, y, list(data.COVERTYPE_NAMES))


class TestLoader(unittest.TestCase):
    def test_reads_the_gzip_as_uci_ships_it(self):
        ds = fake_covertype(40)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "covertype"
            folder.mkdir()
            table = np.column_stack([ds.X.astype(int), ds.y])
            with gzip.open(folder / data.COVERTYPE_FILE, "wt") as f:
                f.write("\n".join(",".join(map(str, r)) for r in table) + "\n")
            got = data.load("covertype", data_dir=Path(tmp))
            np.testing.assert_array_equal(got.X, ds.X)
            np.testing.assert_array_equal(got.y, ds.y)
            self.assertEqual((got.X.shape, len(got.feature_names)), ((40, 54), 54))
            with gzip.open(folder / data.COVERTYPE_FILE, "wt") as f:
                f.write("1,2,3\n")
            with self.assertRaises(ValueError):
                data.load("covertype", data_dir=Path(tmp))

    def test_missing_file_message_and_other_lists_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "covtype.data.gz"):
                data.load("covertype", data_dir=Path(tmp))
        self.assertNotIn("covertype", data.UCI)          # earlier scripts' dataset lists
        self.assertNotIn("covertype", data.BUILTIN)

    def test_real_file_if_present(self):
        path = data.DATA_DIR / "covertype" / data.COVERTYPE_FILE
        if not path.exists():
            self.skipTest("covertype not downloaded")
        ds = data.load("covertype")
        self.assertEqual(ds.X.shape, (581_012, 54))
        self.assertEqual(sorted(np.unique(ds.y)), [1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(np.isin(ds.X[:, 10:], (0, 1)).all())
        self.assertIn("covertype", dd.DATASETS)
        self.assertEqual(dd.status(str(path), dd.DATASETS["covertype"][2][data.COVERTYPE_FILE]), "ok")


class TestScript(unittest.TestCase):
    def test_setup(self):
        self.assertEqual(rk.SEEDS, [170, 171, 172, 173, 174])
        self.assertEqual((rk.THETA, rk.GAMMA, rk.N_EXPLAIN), (0.3, 4.0, 50))
        self.assertEqual(rk.MODELS, ["CFMNN", "10-NN", "Random Forest"])

    def test_prepare_is_the_accuracy_tables_split(self):
        ds = fake_covertype()
        Xtr, ytr, Xte, yte, Xex = rk.prepare(3, ds=ds)
        sp = data.paper_split(ds.y, 3)
        order = np.random.default_rng(3).permutation(sp["train"])
        sc = data.fit_scaler(ds.X[order])
        np.testing.assert_array_equal(Xtr, sc.transform(ds.X[order]))
        np.testing.assert_array_equal(ytr, ds.y[order])
        np.testing.assert_array_equal(yte, ds.y[sp["test"]])
        self.assertEqual(len(Xex), rk.N_EXPLAIN)
        np.testing.assert_array_equal(
            Xex, sc.transform(ds.X[np.random.default_rng(3).permutation(sp["test"])[:rk.N_EXPLAIN]]))
        self.assertEqual((len(ytr), len(yte)), (1200, 300))
        small = rk.prepare(3, subsample=0.2, ds=ds)
        self.assertEqual(len(small[1]) + len(small[3]), 300)

    def test_rows_and_summary(self):
        ds = fake_covertype()
        with mock.patch.object(rk, "THETA", 0.5), mock.patch.object(rk, "N_EXPLAIN", 4):
            rows = [r for seed in (190, 191) for r in rk.run_one("covertype", seed, 1.0, ds)]
            Xtr, ytr, Xte, yte, _ = rk.prepare(190, ds=ds)
            m = CFMNN(theta=0.5).fit(Xtr, ytr)
        self.assertEqual([r["model"] for r in rows[:3]], rk.MODELS)
        self.assertTrue(all(list(r) == rk.COLUMNS for r in rows))
        c = rows[0]
        self.assertEqual(c["boxes"], m.n_boxes_)
        self.assertAlmostEqual(c["test_accuracy"], 100 * np.mean(m.predict(Xte) == yte))
        self.assertEqual(c["train_errors"], training_error_ties(m, Xtr, ytr)["train_errors"])
        self.assertGreater(c["train_errors"], 0)
        self.assertEqual(c["train_errors_tied_inside_both"], c["train_errors"])   # Proposition 1
        self.assertEqual(c["pace_shap_mode"], "sampled")
        self.assertGreater(c["pace_flip_ms"], 0)
        self.assertTrue(np.isnan(rows[1]["boxes"]))
        with tempfile.TemporaryDirectory() as tmp:
            text = rk.summarise(pd.DataFrame(rows), Path(tmp))
            self.assertIn("Descriptive: no comparison is tested", text)
            self.assertIn("(100.00%)", text)
            for m_name in rk.MODELS:
                self.assertIn(f"| {m_name} |", text)
            self.assertTrue((Path(tmp) / "summary.md").exists())

    def test_peak_memory_is_a_number(self):
        self.assertGreater(rk.peak_memory_mb(), 1)


if __name__ == "__main__":
    unittest.main()
