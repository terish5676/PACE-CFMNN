"""Recompute every number the write-up takes from Run M (Simpson's classifier trained from scratch).

Run M: results/simpson_fmm/exact.csv (the exact check: PACE-SHAP against shap's ExactExplainer on
SimpsonFMM, 6 datasets with at most 12 features, seeds 220-229, 5 points each; PASS if the largest
difference is at most 1e-9) and results/simpson_fmm/runs.csv (SimpsonFMM beside CFMNN on the 13
accuracy datasets, θ = 0.3, γ = 4; descriptive, no test); home PC, 12 jobs. Means over seeds;
shares pooled over seeds; differences are SimpsonFMM minus CFMNN. The CSVs are read with
float_precision="round_trip". The cause of the one large exact-check difference is in
check_exact_isclose_simpson.py (post hoc).

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_simpson.py > research_audit/scripts/claims_simpson_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, section

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

rd = lambda p: pd.read_csv(p, float_precision="round_trip")  # noqa: E731
E = rd(RES / "simpson_fmm" / "exact.csv")
M = rd(RES / "simpson_fmm" / "runs.csv")
BAD = []
ORDER = ["S1", "S2", "S3", "wine", "breast_cancer", "digits", "thyroid", "shuttle", "skin", "magic",
         "letter", "landsat", "page_blocks"]
SIM, CF = "Simpson-scratch", "CFMNN"


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def r(x, nd=1):
    return round(float(x), nd)


section("Exact check: PACE-SHAP against ExactExplainer on SimpsonFMM")
check("rows", len(E), 60)
check("datasets", sorted(E.dataset.unique().tolist()), sorted(["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]))
check("seeds", sorted(E.seed.unique().tolist()), list(range(220, 230)))
check("points and explainer rows per task", (sorted(E.n.unique().tolist()), sorted(E.background_rows.unique().tolist())), ([5], [200]))
worst = E.loc[E.max_abs_diff.idxmax()]
print(E.groupby("dataset")[["boxes", "contractions", "max_abs_diff"]].agg({"boxes": "mean", "contractions": "mean", "max_abs_diff": "max"}).to_string())
check("largest difference (2 significant figures)", f"{worst.max_abs_diff:.1e}", "1.4e-10")
check("... is at most the plan's 1e-9 (PASS)", bool(E.max_abs_diff.max() <= 1e-9), True)
check("... and comes from one task", (worst.dataset, int(worst.seed), int((E.max_abs_diff > 1e-14).sum())), ("S1", 222, 1))
check("the other 59 tasks: largest difference", f"{E[E.max_abs_diff <= 1e-14].max_abs_diff.max():.1e}", "3.1e-16")
t = E.groupby("dataset")[["pace_ms_per_explanation", "shap_exact_ms_per_explanation"]].mean()
ratio = t.shap_exact_ms_per_explanation / t.pace_ms_per_explanation
print((t.assign(times=ratio)).round(1).to_string())
check("ExactExplainer's time over PACE-SHAP's during the run (12 jobs), range over datasets", (r(ratio.min(), 0), r(ratio.max(), 0)), (11.0, 30.0))

section("SimpsonFMM beside CFMNN on the 13 accuracy datasets")
check("rows", len(M), 260)
check("seeds", sorted(M.seed.unique().tolist()), list(range(220, 230)))
check("datasets", sorted(M.dataset.unique().tolist()), sorted(ORDER))
acc = M.pivot_table(index=["dataset", "seed"], columns="model", values="test_accuracy")
f1 = M.pivot_table(index=["dataset", "seed"], columns="model", values="test_macro_f1")
rows = []
for d in ORDER:
    a, f = acc.loc[d], f1.loc[d]
    g = M[M.dataset == d]
    s, c = g[g.model == SIM], g[g.model == CF]
    da = a[SIM] - a[CF]
    tie = lambda h: 100 * h.train_errors_tied_inside_both.sum() / h.train_errors.sum() if h.train_errors.sum() else np.nan  # noqa: E731
    rows.append({"dataset": d, "acc CFMNN": a[CF].mean(), "acc Simpson": a[SIM].mean(), "acc diff": da.mean(),
                 "seeds higher": int((da > 0).sum()), "seeds lower": int((da < 0).sum()),
                 "F1 diff": (f[SIM] - f[CF]).mean(), "boxes CFMNN": c.boxes.mean(), "boxes Simpson": s.boxes.mean(),
                 "train err Simpson": int(s.train_errors.sum()), "tie % CFMNN": tie(c), "tie % Simpson": tie(s),
                 "contractions": s.contractions.mean(), "overlap mean": s.overlapping_pairs.mean(),
                 "overlap max": int(s.overlapping_pairs.max()),
                 "tie-rule % CFMNN": 100 * c.test_centroid_decided.sum() / c.n_test.sum(),
                 "tie-rule % Simpson": 100 * s.test_centroid_decided.sum() / s.n_test.sum()})
S = pd.DataFrame(rows).set_index("dataset")
print(S.round(2).to_string())
lower = S[S["acc diff"] < -0.5].sort_values("acc diff")
higher = S[S["acc diff"] > 0.5].sort_values("acc diff", ascending=False)
close = S[S["acc diff"].abs() <= 0.5]
check("lower mean accuracy by more than 0.5 points", lower.index.tolist(), ["shuttle", "letter", "magic", "thyroid"])
check("... by", [r(x) for x in lower["acc diff"]], [-14.0, -8.3, -1.9, -1.0])
check("Shuttle and Letter: CFMNN and Simpson means", [(r(S.loc[d, "acc CFMNN"]), r(S.loc[d, "acc Simpson"])) for d in ("shuttle", "letter")],
      [(99.8, 85.8), (75.3, 67.0)])
check("... lower on every seed on Shuttle and Letter", [int(S.loc[d, "seeds lower"]) for d in ("shuttle", "letter")], [10, 10])
check("higher mean accuracy by more than 0.5 points", higher.index.tolist(), ["skin", "breast_cancer", "page_blocks"])
check("... by", [r(x) for x in higher["acc diff"]], [2.1, 1.1, 0.7])
check("within 0.5 points elsewhere: datasets, largest |difference|", (sorted(close.index.tolist()), r(close["acc diff"].abs().max())),
      (sorted(["S1", "S2", "S3", "wine", "digits", "landsat"]), 0.3))
check("macro-F1 differences on Shuttle, Letter, Thyroid, Skin", [r(S.loc[d, "F1 diff"]) for d in ("shuttle", "letter", "thyroid", "skin")],
      [-10.5, -8.3, -7.1, 3.7])

section("Ties (Proposition 1's form), contraction, overlaps, the tie rule")
has = S[S["train err Simpson"] > 0]
check("CFMNN: every training error a tie wherever it has errors",
      bool((S["tie % CFMNN"].dropna() == 100).all()), True)
check("SimpsonFMM tie share (%), pooled: S2, Breast Cancer, Shuttle, MAGIC",
      [r(S.loc[d, "tie % Simpson"]) for d in ("S2", "breast_cancer", "shuttle", "magic")], [0.0, 0.0, 0.7, 2.1])
check("... Skin, Page Blocks, Landsat, Letter, Thyroid",
      [r(S.loc[d, "tie % Simpson"]) for d in ("skin", "page_blocks", "landsat", "letter", "thyroid")], [37.8, 71.4, 73.9, 98.7, 100.0])
check("datasets where SimpsonFMM has training errors", len(has), 10)
check("contractions: none on Thyroid or Digits (every seed)",
      [int(M[(M.dataset == d) & (M.model == SIM)].contractions.max()) for d in ("thyroid", "digits")], [0, 0])
th = M[M.dataset == "thyroid"].pivot(index="seed", columns="model", values="boxes")
check("Thyroid: same number of boxes as CFMNN on every seed", bool((th[SIM] == th[CF]).all()), True)
check("Thyroid: CFMNN and Simpson accuracy, macro-F1 means",
      [(r(S.loc["thyroid", "acc CFMNN"], 2), r(S.loc["thyroid", "acc Simpson"], 2)), r(S.loc["thyroid", "F1 diff"])],
      [(97.07, 96.04), -7.1])
check("Thyroid: test predictions decided by the tie rule (%), both", [r(S.loc["thyroid", x]) for x in ("tie-rule % CFMNN", "tie-rule % Simpson")], [5.4, 5.4])
check("contractions, Shuttle and MAGIC, means", [r(S.loc[d, "contractions"], 0) for d in ("shuttle", "magic")], [22454.0, 21078.0])
check("pairs still overlapping after training: datasets with any, and the largest count",
      (sorted(S[S["overlap max"] > 0].index.tolist()), int(S["overlap max"].max())), (["landsat", "letter"], 4))

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
