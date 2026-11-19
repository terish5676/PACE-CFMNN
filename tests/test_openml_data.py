"""The OpenML loaders (Runs P9 and P10): the ARFF reader on a toy file, and the downloaded files'
counts against the frozen rule (results/heldout/RULE.md)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pace_cfmnn import data

TOY = """% a comment
@relation 'toy'
@attribute 'a b' numeric
@attribute c REAL
@attribute day {1,2,3}
@attribute 'class' {'w','b'}

@data
0.5,1,2,"b"
0.25, 2, 3, 'w'
"""


class TestArffReader(unittest.TestCase):
    def test_toy_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "phoneme"
            folder.mkdir()
            (folder / data.OPENML_FILES["phoneme"]).write_text(TOY, encoding="utf-8")
            ds = data.load_openml("phoneme", data_dir=Path(tmp))
        np.testing.assert_array_equal(ds.X, [[0.5, 1, 2], [0.25, 2, 3]])
        np.testing.assert_array_equal(ds.y, [1, 0])          # declared order: w = 0, b = 1
        self.assertEqual(ds.feature_names, ["a b", "c", "day"])
        self.assertEqual(ds.info["nominal_features"], ["day"])

    def test_missing_values_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.arff"
            path.write_text(TOY.replace("0.25,", "?,"), encoding="utf-8")
            with self.assertRaises(ValueError):
                data.read_arff(path)

    def test_missing_file_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                data.load_openml("wilt", data_dir=Path(tmp))


# (rows, features, classes, smallest class), as listed in results/heldout/RULE.md and on OpenML.
EXPECTED = {"phoneme": (5404, 5, 2, 1586), "wilt": (4839, 5, 2, 261),
            "mfeat_morphological": (2000, 6, 10, 200), "jungle_chess": (44819, 6, 3, 4335),
            "electricity": (45312, 8, 2, 19237)}


class TestDownloadedFiles(unittest.TestCase):
    def test_counts_match_the_rule(self):
        for name, (n, d, k, smallest) in EXPECTED.items():
            with self.subTest(name=name):
                if not (data.DATA_DIR / name / data.OPENML_FILES[name]).exists():
                    self.skipTest(f"{name} not downloaded")
                ds = data.load(name)
                counts = np.bincount(ds.y)
                self.assertEqual(ds.X.shape, (n, d))
                self.assertEqual((len(counts), int(counts.min())), (k, smallest))
                self.assertFalse(np.isnan(ds.X).any())

    def test_electricity_is_an_unbroken_daily_sequence(self):
        if not (data.DATA_DIR / "electricity" / data.OPENML_FILES["electricity"]).exists():
            self.skipTest("electricity not downloaded")
        ds = data.load("electricity")
        self.assertTrue(ds.info["time_ordered"])
        day, period = ds.X[:, 1].astype(int), ds.X[:, 2]
        starts = np.r_[0, np.flatnonzero(np.diff(period) < 0) + 1]
        self.assertEqual(len(starts), 944)                          # 944 days of 48 half-hours
        self.assertTrue(np.all(np.diff(np.r_[starts, len(day)]) == 48))
        self.assertTrue(np.all((day[starts][1:] - day[starts][:-1]) % 7 == 1))   # one day apart

    def test_heldout_is_the_rule_list(self):
        self.assertEqual(set(data.HELDOUT), {"phoneme", "wilt", "mfeat_morphological", "jungle_chess"})


if __name__ == "__main__":
    unittest.main()
