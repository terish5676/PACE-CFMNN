"""Recompute every number the write-up takes from Run P (where PACE-SHAP's speed comes from).

Run P: results/speed_source/runs.csv (PACE-SHAP, PACE-SHAP without pruning and shap's
ExactExplainer on the 6 RQ1 datasets with at most 12 features, seeds 240-242, 2 points per seed,
200 explainer rows, one task at a time), home PC, run alone. Descriptive, no test. Per dataset,
means over seeds: structure = ExactExplainer / no pruning, pruning = no pruning / PACE-SHAP,
total = ExactExplainer / PACE-SHAP. The CSV is read with float_precision="round_trip".

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_speed_source.py > research_audit/scripts/claims_speed_source_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, section

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)

S = pd.read_csv(RES / "speed_source" / "runs.csv", float_precision="round_trip")
BAD = []
ORDER = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
PACE, NOPR, EXACT = "PACE-SHAP", "PACE-SHAP, no pruning", "shap ExactExplainer"


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def r(x, nd=1):
    return round(float(x), nd)


section("Rows and the check")
check("rows", len(S), 6 * 3 * 3)
check("datasets", sorted(S.dataset.unique()), sorted(ORDER))
check("seeds", sorted(S.seed.unique().tolist()), [240, 241, 242])
check("points and explainer rows", (sorted(S.n.unique().tolist()), sorted(S.background_rows.unique().tolist())), ([2], [200]))
check("d per dataset", S.groupby("dataset").d.first().to_dict(),
      {"S1": 10, "S2": 10, "S3": 10, "shuttle": 9, "magic": 10, "page_blocks": 10})
check("no pruning agrees with PACE-SHAP within 1e-12", bool(S[S.arm == NOPR].max_abs_diff.max() <= 1e-12), True)
check("ExactExplainer agrees within 1e-9", bool(S[S.arm == EXACT].max_abs_diff.max() <= 1e-9), True)
check("largest difference over both arms (2 s.f.)", f"{S[S.arm != PACE].max_abs_diff.max():.1e}", "2.5e-16")

section("The split of the speed-up (means over seeds)")
ms = S.pivot_table(index="dataset", columns="arm", values="ms_per_explanation", aggfunc="mean").loc[ORDER]
f = pd.DataFrame({"structure": ms[EXACT] / ms[NOPR], "pruning": ms[NOPR] / ms[PACE], "total": ms[EXACT] / ms[PACE]})
f["skipped"] = S[S.arm == PACE].groupby("dataset").share_pairs_skipped.mean()
f["boxes"] = S[S.arm == PACE].groupby("dataset").boxes.mean()
print(pd.concat([ms.round(1), f.round(3)], axis=1).to_string())
check("structure factor per dataset", [r(x) for x in f.structure], [7.3, 7.4, 7.3, 10.7, 7.9, 8.6])
check("pruning factor per dataset", [r(x) for x in f.pruning], [2.0, 2.0, 2.0, 1.8, 2.9, 2.6])
check("total per dataset", [r(x) for x in f.total], [14.3, 14.5, 14.7, 18.8, 22.7, 22.7])
check("structure range", (r(f.structure.min()), r(f.structure.max())), (7.3, 10.7))
check("pruning range", (r(f.pruning.min()), r(f.pruning.max())), (1.8, 2.9))
check("total range", (r(f.total.min()), r(f.total.max())), (14.3, 22.7))
check("share of pairs skipped, range", (round(f.skipped.min(), 3), round(f.skipped.max(), 3)), (0.623, 0.759))
check("structure exceeds pruning on every dataset", bool((f.structure > f.pruning).all()), True)
check("S1 ms: ExactExplainer, no pruning, PACE-SHAP", [r(ms.loc["S1", a]) for a in (EXACT, NOPR, PACE)], [27819.8, 3823.1, 1948.7])
check("ms per point, PACE-SHAP, range", (r(ms[PACE].min()), r(ms[PACE].max())), (8.7, 1957.4))

section("Verdict")
print("ALL OK" if not BAD else f"{len(BAD)} BAD: {BAD}")
