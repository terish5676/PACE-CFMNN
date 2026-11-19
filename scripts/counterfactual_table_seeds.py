"""The paper's counterfactual table, tested over seeds (review issue M6; post hoc).

The paper's first counterfactual table tested PACE against DiCE over points
pooled from 3 models, which overstates significance. This script rebuilds the
table from B1's run (results/extensions_v1/counterfactuals/runs.csv: seeds
40-49, 30 test points per seed, 11 datasets). For each dataset and seed it
averages each method over the points where both PACE and DiCE found a valid
counterfactual, as review_reanalysis.m6 does. It then tests PACE against DiCE
over the 10 seeds: two-sided Wilcoxon, Holm across the 11 datasets
(review_reanalysis.compare).

Writes results/review_v1/counterfactual_table.md.

    python -B scripts/counterfactual_table_seeds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_reanalysis import OUT, REAL, RES, RQ1, compare  # noqa: E402

NAMES = {"wine": "Wine", "breast_cancer": "Breast Cancer", "thyroid": "Thyroid", "shuttle": "Shuttle",
         "magic": "MAGIC", "letter": "Letter", "landsat": "Landsat", "page_blocks": "Page Blocks"}
COLUMNS = [("l1", "Size of change"), ("n_changed", "Features changed"),
           ("plausibility", "Distance to real data of the target class"), ("ms", "Time (ms)")]


def per_seed() -> pd.DataFrame:
    """One row per dataset, seed and method: means over the points where both are valid."""
    b = pd.read_csv(RES / "extensions_v1" / "counterfactuals" / "runs.csv")
    rows = []
    for (ds, s), g in b[b.method.isin(["PACE", "DiCE"])].groupby(["dataset", "seed"]):
        P, D = (g[g.method == m].set_index("point") for m in ("PACE", "DiCE"))
        both = P.index[P.valid.eq(True) & D.reindex(P.index).valid.eq(True)]
        for m, X in (("PACE", P), ("DiCE", D)):
            # Time is the median per counterfactual, as in B1's summary (DiCE has slow outliers).
            rows.append({"dataset": ds, "seed": s, "method": m, "n_both": len(both),
                         **{k: X.loc[both, k].median() if k == "ms" else X.loc[both, k].mean()
                            for k, _ in COLUMNS}})
    return pd.DataFrame(rows)


def fmt(v: float, metric: str) -> str:
    if metric == "ms":
        return f"{v:.1f}" if v < 100 else f"{v:,.0f}"
    return f"{v:.2f}"


def main() -> None:
    q = per_seed()
    tests = {m: compare(q, "PACE", "DiCE", m, True, RQ1).set_index("dataset") for m, _ in COLUMNS}
    L = ["# Counterfactuals: PACE vs DiCE, tested over seeds", "",
         "From B1 (`extensions_v1/counterfactuals/runs.csv`, seeds 40–49, 30 test points per seed), on the "
         "points where both methods found a valid counterfactual. Each cell is PACE / DiCE: the mean over "
         "seeds of each seed's mean (for time, each seed's median per counterfactual). Several runs shared "
         "the CPU, so compare times within a row only. Bold: significant over seeds (two-sided Wilcoxon, "
         "Holm across the 11 datasets, p < 0.05). Post hoc. Made by `scripts/counterfactual_table_seeds.py`.",
         "",
         "| Dataset | " + " | ".join(t for _, t in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for ds in RQ1:
        cells = []
        for m, _ in COLUMNS:
            r = tests[m].loc[ds]
            c = f"{fmt(r.a, m)} / {fmt(r.b, m)}"
            cells.append(f"**{c}**" if r.p_holm < 0.05 else c)
        L.append(f"| {NAMES.get(ds, ds)} | " + " | ".join(cells) + " |")
    L.append("")
    for m, title in COLUMNS:
        t = tests[m]
        every, real = t.verdict.value_counts(), t.loc[REAL].verdict.value_counts()
        better = [NAMES.get(d, d) for d in t.index if t.loc[d, "verdict"] == "better"]
        L.append(f"- {title}: PACE better on {every.get('better', 0)} of 11 ({real.get('better', 0)} of the 8 "
                 f"real datasets), level on {every.get('level', 0)}, worse on {every.get('worse', 0)}. "
                 f"Better on: {', '.join(better) or 'none'}.")
    ratio = tests["ms"].b / tests["ms"].a
    L.append(f"- Speed: DiCE takes {ratio.min():.0f}–{ratio.max():,.0f}× as long as PACE.")
    n = q[q.method == "PACE"].n_both
    L.append(f"- Points per seed where both are valid: {n.min()}–{n.max()} of 30.")
    text = "\n".join(L) + "\n"
    (OUT / "counterfactual_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
