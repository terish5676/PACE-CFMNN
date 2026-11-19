"""Recompute every number the write-up takes from Run J (sampled baselines above 12 features).

Run J: results/sampled_baselines/runs.csv and timing.csv (pre-registered, seeds 160-169; the
timing pass on seeds 160-162, alone; home PC). P1, P2 and the insertion secondaries are redone
with ``common.paired`` (paired two-sided Wilcoxon over the 10 seeds, Holm across the 5
datasets).

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_sampled.py > research_audit/scripts/claims_sampled_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")

J = pd.read_csv(RES / "sampled_baselines" / "runs.csv")
T = pd.read_csv(RES / "sampled_baselines" / "timing.csv")
DS = ["wine", "breast_cancer", "thyroid", "letter", "landsat"]
P, PERM, KER = "PACE-SHAP", "PermutationSHAP", "KernelSHAP-paired"
BAD = []


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def test(b, col, lower):
    out = paired(J, P, b, col, DS, lower_better=lower).set_index("unit")
    print(f"\n{P} vs {b} ({col}):")
    print(out[["a", "b", "diff", "p", "p_holm", "a_better_seeds", "verdict"]].round(5).to_string())
    return out


def verdicts(out):
    return {v: sorted(out.index[out.verdict == v]) for v in ("better", "worse") if (out.verdict == v).any()}


section("Completeness and budget")
check("rows", len(J), 200)
check("timing rows", len(T), 45)
check("seeds", sorted(J.seed.unique().tolist()), list(range(160, 170)))
check("features", J.groupby("dataset").d.first()[DS].tolist(), [13, 30, 21, 16, 36])
base = J[J.method.isin([PERM, KER])]
check("every baseline within its budget", bool((base.rows_per_explanation <= base.budget_rows).all()), True)

section("P1, P2 (primary): deletion margin AUC (lower is more faithful)")
p1 = test(PERM, "margin_auc", True)
p2 = test(KER, "margin_auc", True)
check("P1 verdicts", verdicts(p1), {"better": ["landsat"]})
check("P2 verdicts", verdicts(p2), {"better": ["landsat"]})
check("P1, P2 Landsat differences", (round(float(p1.loc["landsat", "diff"]), 4), round(float(p2.loc["landsat", "diff"]), 4)),
      (-0.0045, -0.0057))
check("largest |difference| on the other 4", round(float(pd.concat([p1["diff"], p2["diff"]]).drop("landsat").abs().max()), 4),
      0.0006)

section("Secondary: insertion margin AUC (higher is more faithful)")
i1 = test(PERM, "insertion_margin_auc", False)
i2 = test(KER, "insertion_margin_auc", False)
check("insertion vs PermutationSHAP", verdicts(i1), {"better": ["wine"]})
check("insertion vs KernelSHAP-paired", verdicts(i2), {"better": ["landsat", "thyroid"]})

section("Error against exact values (Wine, Letter; first 10 points per seed)")
err = J[J.dataset.isin(["wine", "letter"]) & (J.method != "Random")].pivot_table(
    index="dataset", columns="method", values="rel_l1_error")
print(err.round(4).to_string())
check("relative L1 error, Wine (PACE-SHAP, Permutation, Kernel)",
      [round(float(err.loc["wine", m]), 3) for m in (P, PERM, KER)], [0.010, 0.029, 0.025])
check("relative L1 error, Letter", [round(float(err.loc["letter", m]), 3) for m in (P, PERM, KER)], [0.014, 0.046, 0.049])
ratio = pd.concat([err[P] / err[PERM], err[P] / err[KER]])
check("PACE-SHAP's error as a share of the baselines'", (round(float(ratio.min()), 2), round(float(ratio.max()), 2)),
      (0.28, 0.40))

section("Top-3 stability under a new seed")
st = J.pivot_table(index="dataset", columns="method", values="stability_top3").loc[DS]
print(st.round(3).to_string())
check("PACE-SHAP the steadiest on every dataset", bool((st[P] > st[[PERM, KER]].max(axis=1)).all()), True)
check("Landsat (PACE-SHAP, Permutation, Kernel)", [round(float(st.loc["landsat", m]), 2) for m in (P, PERM, KER)],
      [0.95, 0.76, 0.71])

section("Timing pass (seeds 160-162, one task at a time): milliseconds per explanation")
tm = T.pivot_table(index="dataset", columns="method", values="ms_per_explanation").loc[DS]
print(tm.round(1).to_string())
check("PACE-SHAP ms", [round(float(tm.loc[d, P])) for d in DS], [7, 84, 60, 524, 636])
r = pd.concat([tm[PERM] / tm[P], tm[KER] / tm[P]])
check("baselines' time as a multiple of PACE-SHAP's", (round(float(r.min()), 1), round(float(r.max()), 1)), (3.8, 9.6))

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
