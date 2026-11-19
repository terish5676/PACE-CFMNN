"""Recompute every number the paper takes from the follow-up tests (runs E, A, B, C, D).

The runs, their plans and their own summaries are in results/fast_gate_check/,
results/repair_sensitivity/, results/cf_robustness/, results/rq1_more_points/ and
results/conditional_settings/. The paper's text is in §7 ("Follow-up tests of the
limitations"), Limitations, Deviations and Settings (doc rev 325). This script redoes each
test with ``common.paired`` (paired two-sided Wilcoxon over seeds 110-119, Holm across
datasets) instead of the runs' own summary code.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_followup.py > research_audit/scripts/claims_followup_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, paired, section, tally

sys.stdout.reconfigure(encoding="utf-8")

E = pd.read_csv(RES / "fast_gate_check" / "runs.csv")
A = pd.read_csv(RES / "repair_sensitivity" / "runs.csv")
B = pd.read_csv(RES / "cf_robustness" / "runs.csv")
C = pd.read_csv(RES / "rq1_more_points" / "runs.csv")
D = pd.read_csv(RES / "conditional_settings" / "runs.csv")


def check(label, got, want):
    ok = got == want
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (paper: {want})"))


section("Completeness")
for name, df, n in (("E", E, 60), ("A", A, 60 * 9), ("B", B, 110 * 30 * 7), ("C", C, 110 * 11),
                    ("D", D, 60 * 4 * 2)):
    check(f"run {name} rows", len(df), n)

section("E: the cached gate")
check("edit decisions replayed", int(E.n_edits.sum()), 14827)
check("edit decisions that differ", int(E.edit_mismatches.sum()), 0)
check("runs equal to the committed macro-F1", int((E.macro_f1_abs_diff <= 1e-12).sum()), 60)
s = E[E.standard_run.astype(bool)]
check("seed-50 standard runs equal to cached", int(s.standard_equals_cached.astype(bool).sum()), 6)
r = s.standard_s / s.cached_s
print(f"speed-up per dataset {r.min():.1f}-{r.max():.1f}; total {s.standard_s.sum() / s.cached_s.sum():.1f}"
      "  (paper: 2-22 times, 12 in total)")

section("A: budgets and gates (macro-F1 vs SR-default)")
ds = list(A.dataset.unique())
for arm in ["SR-b100", "SR-b1000", "SR-ball", "SR-gate-none", "SR-gate-strict", "SR-gate-loose",
            "SR-gate-cumulative"]:
    out = paired(A, arm, "SR-default", "test_macro_f1", ds, method_col="arm")
    print(f"{arm}: {tally(out)}; diffs {dict(zip(out.unit, out['diff'].round(2)))}")
m = A.groupby(["dataset", "arm"]).test_min_class_recall.mean()
print("MAGIC lowest recall: B0 %.1f, SR-default %.1f, cumulative %.1f  (paper: 69.6, 68.0, 70.8)"
      % (m["magic", "B0"], m["magic", "SR-default"], m["magic", "SR-gate-cumulative"]))
c = A[A.arm == "SR-default"]
check("certified edits", int(c.cert_edits.sum()), 6272)
for k in ("loose", "tight", "centroid", "flip", "decomposition"):
    # the size and precision columns are already in percent
    print(f"{k}: misses {int(c[f'cert_{k}_misses'].sum())}, size {c[f'cert_{k}_size_share'].mean():.2f}%"
          f", precision {c[f'cert_{k}_precision'].mean():.2f}%")
print("(paper: centroid 0.8% / 38%, tight 6.3% / 6%, decomposition 5.0% / 99%, flip 55% / 23%; 0 misses)")

section("B: counterfactual robustness (per dataset and seed, mean over points)")
per = B.groupby(["dataset", "seed", "method"])[["r1", "r3", "l1"]].mean().reset_index()
bds = list(B.dataset.unique())
for a in ("PACE", "PACE-sparse"):
    print(a, "r1:", tally(paired(per, a, "DiCE random", "r1", bds, lower_better=True)))
    print(a, "r3:", tally(paired(per, a, "DiCE random", "r3", bds)))
mm = B.groupby(["dataset", "method"]).r1.mean().unstack()   # mean over points and seeds
for a, want in (("PACE", "2.5-13"), ("PACE-sparse", "1.5-10")):
    ratio = mm["DiCE random"] / mm[a]
    print(f"DiCE random r1 / {a} r1: {ratio.min():.2f}-{ratio.max():.2f}  (paper: {want})")
l1 = per.groupby(["dataset", "method"]).l1.mean().unstack()
print("genetic and kdtree L1 above PACE on every dataset:",
      bool((l1["DiCE genetic"] > l1["PACE"]).all() and (l1["DiCE kdtree"] > l1["PACE"]).all()))

section("C: RQ1 with 200 points")
cds = list(C.dataset.unique())
for b in ("KernelSHAP", "KernelSHAP-same"):
    for col, low in (("margin_auc", True), ("insertion_margin_auc", False),
                     ("margin_auc_untied", True), ("insertion_margin_auc_untied", False),
                     ("margin_auc_raw", True), ("insertion_margin_auc_raw", False)):
        print(f"PACE-SHAP vs {b}, {col}: {tally(paired(C, 'PACE-SHAP', b, col, cds, lower_better=low))}")

section("D: conditional settings")
dds = list(D.dataset.unique())
for st in D.setting.unique():
    sub = D[D.setting == st]
    for col, low in (("cond_margin_auc", True), ("cond_insertion_margin_auc", False)):
        print(f"{st}, {col}: {tally(paired(sub, 'PACE-SHAP-cond', 'KernelSHAP-cond', col, dds, lower_better=low))}")
g = D.groupby(["dataset", "setting"])[["eff_rows_large", "off_share"]].mean()
for st in ("sigma=0.05, rows=200", "sigma=0.4, rows=200"):
    e = g.xs(st, level="setting")
    print(f"{st}: effective rows (large) {e.eff_rows_large.min():.1f}-{e.eff_rows_large.max():.1f}; "
          f"off-data Letter {100 * e.off_share['letter']:.0f}%, Shuttle {100 * e.off_share['shuttle']:.0f}%")
print("(paper: 1-3 rows at 0.05, 27-120 at 0.4; Letter 10% -> 34%, Shuttle 49% -> 67%)")
