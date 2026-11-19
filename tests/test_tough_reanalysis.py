"""Tough review D1: the Nadeau-Bengio corrected resampled t-test in the reanalysis script."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tough_review_reanalysis import nb_p, verdict  # noqa: E402


class TestNadeauBengio(unittest.TestCase):
    def test_ratio_zero_is_the_paired_t_test(self):
        rng = np.random.default_rng(0)
        d = rng.normal(0.3, 1.0, size=10)
        self.assertAlmostEqual(nb_p(d, 0.0), stats.ttest_1samp(d, 0.0).pvalue, places=12)

    def test_the_correction_scales_the_variance(self):
        rng = np.random.default_rng(1)
        d = rng.normal(0.5, 1.0, size=10)
        J, r = len(d), 0.25
        t = d.mean() / np.sqrt((1 / J + r) * d.var(ddof=1))
        self.assertAlmostEqual(nb_p(d, r), 2 * stats.t.sf(abs(t), J - 1), places=12)
        self.assertGreater(nb_p(d, r), nb_p(d, 0.0))
        # 80/20 split, 10 seeds: the variance is 3.5 times the plain test's
        self.assertAlmostEqual((1 / J + r) / (1 / J), 3.5)

    def test_degenerate_cases(self):
        self.assertEqual(nb_p(np.zeros(10), 0.25), 1.0)
        self.assertEqual(nb_p(np.full(10, 0.2), 0.25), 0.0)
        self.assertEqual(nb_p(np.array([0.1]), 0.25), 1.0)

    def test_verdict_direction(self):
        v = verdict([0.01, 0.01, 0.2], [-1.0, 1.0, -1.0], np.array([True, True, True]))
        self.assertEqual(list(v), ["better", "worse", "not significant"])


if __name__ == "__main__":
    unittest.main()
