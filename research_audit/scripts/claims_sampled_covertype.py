"""Recompute every number the write-up takes from Run N (Covertype in the sampled-baseline comparison).

Run N: results/sampled_baselines_covertype/runs.csv (seeds 230-234, 50 points each, 5 jobs) and
timing.csv (timing pass alone, seeds 230-232, 10 points each); home PC. Descriptive: no test.
Means and standard deviations over the 5 seeds; per-seed differences PACE-SHAP minus each
baseline (deletion: negative favours PACE-SHAP; insertion: positive favours PACE-SHAP). The CSVs
are read with float_precision="round_trip". For comparison, Run J's timing pass
(results/sampled_baselines/timing.csv) gives the samplers' time ratios on the other 5 datasets.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_sampled_covertype.py > research_audit/scripts/claims_sampled_covertype_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, section

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

rd = lambda p: pd.read_csv(p, float_precision="round_trip")  # noqa: E731
N = rd(RES / "sampled_baselines_covertype" / "runs.csv")
T = rd(RES / "sampled_baselines_covertype" / "timing.csv")
J = rd(RES / "sampled_baselines" / "timing.csv")
BAD = []
METHODS = ["PACE-SHAP", "PermutationSHAP", "KernelSHAP-paired", "Random"]
BASE = ["PermutationSHAP", "KernelSHAP-paired"]
EPS = 1e-12   # a per-seed difference smaller than this is rounding: counted as level


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def r(x, nd=4):
    return round(float(x), nd)


section("Completeness and setup")
check("rows", len(N), 20)
check("timing rows", len(T), 9)
check("seeds", sorted(N.seed.unique().tolist()), list(range(230, 235)))
check("timing seeds", sorted(T.seed.unique().tolist()), [230, 231, 232])
check("features, points, explainer rows", (sorted(N.d.unique().tolist()), sorted(N.n.unique().tolist()),
      sorted(N.background_rows.unique().tolist())), ([54], [50], [200]))
check("budget (model rows per explanation)", sorted(N.budget_rows.unique().tolist()), [352000])
check("every method within the budget", bool((N.rows_per_explanation <= N.budget_rows).all()), True)
check("no exact reference", sorted(N.n_exact.unique().tolist()), [0])
check("boxes, range over seeds", (int(N.boxes.min()), int(N.boxes.max())), (109, 121))

section("Means and standard deviations over seeds")
m = N.groupby("method")[["margin_auc", "insertion_margin_auc", "stability_top3", "rows_per_explanation"]].agg(["mean", "std"]).loc[METHODS]
print(m.round(4).to_string())
check("deletion margin AUC, means (3 dp)", [r(m.loc[x, ("margin_auc", "mean")], 3) for x in METHODS],
      [-0.215, -0.215, -0.213, -0.093])
check("deletion margin AUC, means (4 dp)", [r(m.loc[x, ("margin_auc", "mean")]) for x in METHODS[:3]],
      [-0.2151, -0.2153, -0.2128])
check("insertion margin AUC, means", [r(m.loc[x, ("insertion_margin_auc", "mean")]) for x in METHODS[:3]],
      [0.0568, 0.056, 0.0514])
check("top-3 stability, means", [r(m.loc[x, ("stability_top3", "mean")], 3) for x in METHODS[:3]], [0.98, 0.944, 0.978])
rows = {x: r(m.loc[x, ("rows_per_explanation", "mean")], 0) for x in METHODS[:3]}
check("model rows per explanation, means", list(rows.values()), [352000.0, 114777.0, 351801.0])
check("PermutationSHAP's share of the budget (%)", r(100 * rows["PermutationSHAP"] / 352000, 1), 32.6)

section("Per seed, PACE-SHAP minus each baseline")
for col, sign, name in (("margin_auc", -1, "deletion"), ("insertion_margin_auc", 1, "insertion")):
    p = N.pivot(index="seed", columns="method", values=col)
    d = pd.DataFrame({b: p["PACE-SHAP"] - p[b] for b in BASE})
    print(f"{name}:\n{d.round(5).to_string()}")
    for b in BASE:
        ahead = int((sign * d[b] > EPS).sum())
        behind = int((sign * d[b] < -EPS).sum())
        print(f"  against {b}: PACE-SHAP ahead on {ahead} seeds, behind on {behind}, level on {5 - ahead - behind}; "
              f"mean {d[b].mean():+.5f}; largest |difference| {d[b].abs().max():.4f}")
    if name == "deletion":
        check("deletion: ahead/behind against PermutationSHAP", (int((d.PermutationSHAP < -EPS).sum()), int((d.PermutationSHAP > EPS).sum())), (1, 4))
        check("deletion: ahead/behind against KernelSHAP-paired", (int((d["KernelSHAP-paired"] < -EPS).sum()), int((d["KernelSHAP-paired"] > EPS).sum())), (3, 2))
        check("deletion: largest |difference| per seed", r(d.abs().max().max()), 0.0067)
        check("deletion: mean differences (PermutationSHAP, KernelSHAP-paired)",
              [r(d[b].mean()) for b in BASE], [0.0002, -0.0023])
    else:
        check("insertion: ahead/behind against PermutationSHAP", (int((d.PermutationSHAP > EPS).sum()), int((d.PermutationSHAP < -EPS).sum())), (4, 1))
        check("insertion: ahead/behind/level against KernelSHAP-paired", (int((d["KernelSHAP-paired"] > EPS).sum()), int((d["KernelSHAP-paired"] < -EPS).sum()),
              int((d["KernelSHAP-paired"].abs() <= EPS).sum())), (4, 0, 1))

section("Milliseconds per explanation, timed alone (seeds 230-232, 10 points each)")
t = T.groupby("method").ms_per_explanation.mean()
tp = T.pivot(index="seed", columns="method", values="ms_per_explanation")
ratio = tp.div(tp["PACE-SHAP"], axis=0)
print(t.round(1).to_string(), "\n", ratio.round(2).to_string())
check("PACE-SHAP ms, mean", r(t["PACE-SHAP"], 0), 434.0)
check("times PACE-SHAP's (of the means)", [r(t[b] / t["PACE-SHAP"], 1) for b in BASE], [4.6, 13.9])
check("per-seed ratio ranges", [(r(ratio[b].min(), 1), r(ratio[b].max(), 1)) for b in BASE], [(4.4, 4.7), (13.5, 14.8)])
during = N.groupby("method").ms_per_explanation.mean()
check("during the run (5 jobs), PACE-SHAP ms", r(during["PACE-SHAP"], 0), 979.0)

section("For comparison: Run J's timing pass, times PACE-SHAP's (means per dataset)")
jt = J.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
jr = pd.DataFrame({b: jt[b] / jt["PACE-SHAP"] for b in BASE})
print(jr.round(2).to_string())
check("Run J's range over its 5 datasets and both samplers", (r(jr.min().min(), 1), r(jr.max().max(), 1)), (3.8, 9.6))

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
