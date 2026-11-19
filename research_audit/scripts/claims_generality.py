"""Recompute every number the write-up takes from Run I (generality and the gamma sweep).

Run I: results/generality/runs.csv and exact.csv (pre-registered, seeds 150-159, home PC).
P1 and P2 are redone with ``common.paired`` (paired two-sided Wilcoxon over the 10 seeds, Holm
across the 13 datasets). The tie counts are pooled over seeds, as the plan says. The post hoc
check of the exact check's largest differences is `check_exact_simpson.py`.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_generality.py > research_audit/scripts/claims_generality_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")

R = pd.read_csv(RES / "generality" / "runs.csv")
E = pd.read_csv(RES / "generality" / "exact.csv")
DS = ["S1", "S2", "S3", "wine", "breast_cancer", "digits", "thyroid", "shuttle", "skin", "magic", "letter",
      "landsat", "page_blocks"]
BAD = []


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def test(a, b, col="test_accuracy"):
    out = paired(R, a, b, col, DS, method_col="model").set_index("unit")
    print(f"\n{a} vs {b} ({col}):")
    print(out[["a", "b", "diff", "p", "p_holm", "verdict"]].round(4).to_string())
    return out


def verdicts(out):
    return {v: sorted(out.index[out.verdict == v]) for v in ("better", "worse") if (out.verdict == v).any()}


section("Completeness")
check("runs rows", len(R), 520)
check("exact rows", len(E), 60)
check("seeds", sorted(R.seed.unique().tolist()), list(range(150, 160)))

section("P1 and P2 (primary): test accuracy at gamma 2 and 8 against gamma 4")
p1 = test("gamma=2", "gamma=4")
check("P1 verdicts", verdicts(p1), {"better": ["S1", "digits"]})
check("P1 S1, Digits differences", [round(float(p1.loc[d, "diff"]), 1) for d in ("S1", "digits")], [2.1, 1.0])
p2 = test("gamma=8", "gamma=4")
check("P2 verdicts", verdicts(p2), {"worse": ["S1", "S2", "S3", "digits"]})
check("P2 S1, S2, S3, Digits differences", [round(float(p2.loc[d, "diff"]), 1) for d in ("S1", "S2", "S3", "digits")],
      [-4.3, -2.6, -3.9, -0.9])

section("Proposition 1 at gamma 2, 4 and 8 (ties inside a box of both classes)")
g = R[R.model.str.startswith("gamma")]
errs, tied = int(g.train_errors.sum()), int(g.train_errors_tied_inside_both.sum())
check("training errors, every dataset, seed and gamma", errs, 569239)
check("... that are not ties inside both boxes", errs - tied, 0)
check("... that are not ties (looser count)", int((g.train_errors - g.train_errors_tied).sum()), 0)

section("The Simpson model (CFMNN's boxes after Simpson's contraction) against CFMNN at gamma 4")
s, c = R[R.model == "Simpson"], R[R.model == "gamma=4"]
none = s.groupby("dataset").contractions.apply(lambda v: int((v == 0).sum()))
print("seeds with no contraction:", none.to_dict())
check("seeds with no contraction: Wine, Digits, Thyroid", [int(none[d]) for d in ("wine", "digits", "thyroid")], [9, 10, 10])
check("datasets with a contraction on every seed", sorted(none.index[none == 0]),
      sorted(set(DS) - {"wine", "digits", "thyroid"}))
pool = s.groupby("dataset")[["train_errors", "train_errors_tied"]].sum()
share = (100 * pool.train_errors_tied / pool.train_errors).round(1)
print("Simpson tie share (%), pooled:", share.to_dict())
check("Simpson tie share, S1-S3, Breast Cancer, Shuttle, Skin, MAGIC",
      [float(share[d]) for d in ("S1", "S2", "S3", "breast_cancer", "shuttle", "skin", "magic")],
      [0.0, 0.0, 0.0, 3.6, 0.1, 7.7, 0.2])
check("Simpson tie share, Letter, Landsat, Page Blocks, Thyroid",
      [float(share[d]) for d in ("letter", "landsat", "page_blocks", "thyroid")], [98.6, 70.4, 63.9, 100.0])
low = ["S1", "S2", "S3", "breast_cancer", "shuttle", "skin", "magic"]
check("paper: Simpson tie share range on those 7 (whole %)", (round(float(share[low].min())), round(float(share[low].max()))), (0, 8))
check("paper: P2 range of the 4 losses", (round(float(p2.loc[["S1", "S2", "S3", "digits"], "diff"].max()), 1),
                                          round(float(p2.loc[["S1", "S2", "S3", "digits"], "diff"].min()), 1)), (-0.9, -4.3))
acc = R.pivot_table(index="dataset", columns="model", values="test_accuracy")
print(acc.loc[DS, ["gamma=4", "Simpson"]].round(2).to_string())
check("accuracy, gamma 4 -> Simpson: Shuttle, MAGIC, Page Blocks, Skin",
      [(round(float(acc.loc[d, "gamma=4"]), 1), round(float(acc.loc[d, "Simpson"]), 1))
       for d in ("shuttle", "magic", "page_blocks", "skin")],
      [(99.8, 71.2), (71.2, 58.3), (87.4, 80.5), (94.5, 89.0)])
other = [d for d in DS if d not in ("shuttle", "magic", "page_blocks", "skin")]
check("largest accuracy change elsewhere (points)",
      round(float((acc.loc[other, "Simpson"] - acc.loc[other, "gamma=4"]).abs().max()), 1), 0.5)

section("Exact check: PACE-SHAP on the Simpson model against shap's ExactExplainer")
print(E.sort_values("max_abs_diff", ascending=False)[["dataset", "seed", "contractions", "max_abs_diff"]].head(8).to_string())
check("largest difference", f"{E.max_abs_diff.max():.1e}", "1.6e-10")
check("below the plan's limit 1e-9", bool(E.max_abs_diff.max() <= 1e-9), True)
big = E[E.max_abs_diff > 1e-12]
check("tasks above 1e-12", sorted(zip(big.dataset, big.seed)),
      [("S1", 152), ("S2", 159), ("S3", 150), ("S3", 159), ("magic", 158)])
check("largest of the other 55", f"{E[E.max_abs_diff <= 1e-12].max_abs_diff.max():.1e}", "2.7e-16")

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
