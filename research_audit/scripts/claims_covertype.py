"""Recompute every number the write-up takes from Run K (one large dataset, Covertype).

Run K: results/covertype/runs.csv (descriptive, seeds 170-174, one seed at a time alone; amendment
1 with the file's hash and the smoke slices before the run; home PC). No test: means and
standard deviations over the 5 seeds, tie counts pooled over them.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_covertype.py > research_audit/scripts/claims_covertype_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, section

sys.stdout.reconfigure(encoding="utf-8")

K = pd.read_csv(RES / "covertype" / "runs.csv")
BAD = []


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


section("Completeness")
check("rows", len(K), 15)
check("seeds", sorted(K.seed.unique().tolist()), list(range(170, 175)))
check("full data", K.subsample.unique().tolist(), [1.0])
check("training and test rows", (int(K.n_train.iloc[0]), int(K.n_test.iloc[0])), (464809, 116203))

section("Accuracy and macro-F1 (%), mean and standard deviation over seeds")
m = K.groupby("model")[["test_accuracy", "test_macro_f1", "fit_seconds", "predict_seconds"]].agg(["mean", "std"])
print(m.round(2).to_string())
check("accuracy means (CFMNN, 10-NN, Random Forest)",
      [round(float(m.loc[x, ("test_accuracy", "mean")]), 1) for x in ("CFMNN", "10-NN", "Random Forest")],
      [51.4, 92.7, 95.6])
check("CFMNN accuracy standard deviation", round(float(m.loc["CFMNN", ("test_accuracy", "std")]), 1), 2.7)
check("macro-F1 means", [round(float(m.loc[x, ("test_macro_f1", "mean")]), 1) for x in ("CFMNN", "10-NN", "Random Forest")],
      [39.1, 87.1, 92.7])
check("CFMNN accuracy range over seeds", (round(float(K[K.model == "CFMNN"].test_accuracy.min()), 1),
                                          round(float(K[K.model == "CFMNN"].test_accuracy.max()), 1)), (47.7, 55.1))

section("CFMNN: size, ties, centroid-decided predictions, explanation time")
c = K[K.model == "CFMNN"]
print(c[["seed", "boxes", "fit_seconds", "train_error_pct", "train_errors", "train_errors_tied_inside_both",
         "test_centroid_decided_pct", "pace_flip_ms", "pace_shap_ms", "peak_memory_mb", "task_seconds"]].round(2).to_string())
check("seconds to train, mean", round(float(c.fit_seconds.mean()), 1), 10.3)
check("boxes, mean", round(float(c.boxes.mean())), 119)
check("training error (%), mean", round(float(c.train_error_pct.mean()), 1), 48.5)
check("training errors, pooled", int(c.train_errors.sum()), 1128010)
check("... ties inside a box of both classes", int(c.train_errors_tied_inside_both.sum()), 1128010)
check("test predictions decided by the nearest centroid (%), pooled",
      round(100 * float(c.test_centroid_decided.sum() / c.n_test.sum()), 1), 98.9)
check("test errors among them (%), pooled",
      round(100 * float(c.test_errors_centroid_decided.sum() / c.test_errors.sum()), 1), 99.9)
check("PACE-flip ms, mean", round(float(c.pace_flip_ms.mean()), 1), 1.1)
check("PACE-SHAP ms, mean", round(float(c.pace_shap_ms.mean())), 454)
check("PACE-SHAP mode", c.pace_shap_mode.unique().tolist(), ["sampled"])
check("peak memory (GB), largest", round(float(c.peak_memory_mb.max()) / 1024, 1), 3.0)
check("minutes per seed, mean", round(float(c.task_seconds.mean()) / 60, 1), 2.2)

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
