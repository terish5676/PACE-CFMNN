"""Tough review D4 and D5: tuned fuzzy min-max baselines and the deduplicated Skin variant."""

import importlib.util
import unittest

import numpy as np

from pace_cfmnn import data as D
from pace_cfmnn.fmm_baselines import FMM_MODELS, fit_tuned, make_fmm, n_boxes, tune_theta
from tests.test_refine import problem

HAS_HB = importlib.util.find_spec("hbbrain") is not None


@unittest.skipUnless(HAS_HB, "hyperbox-brain not installed")
class TestFMMBaselines(unittest.TestCase):
    def test_every_model_fits_and_predicts(self):
        X, y = problem(0, n=200, d=3, classes=3)
        for name in FMM_MODELS + ["CFMNN"]:
            m = make_fmm(name, 0.3)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X, y)
                acc = np.mean(np.asarray(m.predict(X)) == y)
            self.assertGreater(n_boxes(m), 0, name)
            self.assertGreater(acc, 0.5, name)

    def test_tuning_picks_from_the_grid(self):
        X, y = problem(1, n=240, d=3, classes=2)
        theta, scores = tune_theta("EFMNN", X, y, seed=0, grid=[0.5, 0.3, 0.1])
        self.assertIn(theta, [0.5, 0.3, 0.1])
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[[0.5, 0.3, 0.1].index(theta)], max(scores))
        model, t2, _ = fit_tuned("CFMNN", X, y, seed=0, grid=[0.5, 0.3, 0.1])
        self.assertIn(t2, [0.5, 0.3, 0.1])


class TestDedup(unittest.TestCase):
    def test_dedup_removes_exact_copies(self):
        X = np.array([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])
        y = np.array([0, 0, 1, 1])
        out = D.dedup(D.Dataset("toy", X, y, ["a", "b"]))
        self.assertEqual(len(out.y), 3)                    # (0.1, 0.2, 1) differs in label: kept
        self.assertEqual(out.info["dedup_rows_before"], 4)

    @unittest.skipUnless((D.DATA_DIR / "skin").exists(), "Skin data not downloaded")
    def test_skin_dedup_size(self):
        self.assertEqual(len(D.load("skin_dedup").y), 51444)


if __name__ == "__main__":
    unittest.main()
