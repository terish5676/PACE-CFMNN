"""Summarise refinement v1: the pre-registered confirmatory run (seeds 10-19).

The run was made by a teammate with the pre-registered protocol
(``results/refinement_confirmatory/PREREGISTRATION.md``) plus two arms added by
``PROTOCOL_AMENDMENT_1.md``. Its ``runs.csv`` is kept unchanged; our own code
reproduces its pre-registered arms exactly (checked on Page Blocks, seed 10).

This script recomputes everything from ``runs.csv`` and ``rounds.csv`` and
writes ``summary_v1.md``:

1. the pre-registered endpoints, R1 (primary) first, then R2, R3 and S1:
   paired two-sided Wilcoxon over the 10 seeds of each dataset, Holm across
   the 6 datasets of each comparison;
2. the descriptive measures the pre-registration lists;
3. every label each arm used, including the acceptance rows the gate reads;
4. an appendix on the two amended arms, with the caveats from the review.

    python scripts/summarise_refinement_v1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "refinement_confirmatory"
DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
NAMES = {"thyroid": "Thyroid", "landsat": "Landsat", "page_blocks": "Page Blocks",
         "magic": "MAGIC", "letter": "Letter", "skin": "Skin"}
OE = "order_ensemble | "
B0, B9 = "B0 no refinement", "B9 refit on all pool labels"
SIMPSON, CMIN = "B-Simpson non-active contraction", OE + "A+B+C-minimal"
PREREG_ARMS = [B0, B9, OE + "A+B", OE + "A+B+C", OE + "A+B+C'-random", OE + "A+B+C'-volume",
               "random | A+B+C"]
SHORT = {B0: "B0 (no refinement)", B9: "B9 (refit on all pool labels)", OE + "A+B": "OE, A+B",
         OE + "A+B+C": "OE, A+B+C (explanation cut)", OE + "A+B+C'-random": "OE, A+B+C′-random",
         OE + "A+B+C'-volume": "OE, A+B+C′-volume", "random | A+B+C": "Random, A+B+C",
         SIMPSON: "B-Simpson (amended)", CMIN: "OE, A+B+C-minimal (amended)"}
ENDPOINTS = [
    ("R1 (primary)", "A+B+C − A+B+C′-volume", OE + "A+B+C", OE + "A+B+C'-volume"),
    ("R2", "A+B+C − A+B", OE + "A+B+C", OE + "A+B"),
    ("R3", "A+B+C − A+B+C′-random", OE + "A+B+C", OE + "A+B+C'-random"),
    ("S1", "order ensemble − random selection (both A+B+C)", OE + "A+B+C", "random | A+B+C"),
]
AMENDED = [
    ("R4", "C-minimal − C′-volume", CMIN, OE + "A+B+C'-volume"),
    ("R5", "C-minimal − C", CMIN, OE + "A+B+C"),
    ("R6", "C-minimal − B-Simpson", CMIN, SIMPSON),
    ("extra", "B-Simpson − B0", SIMPSON, B0),
]


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def compare(piv: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        d = (piv.loc[ds, a] - piv.loc[ds, b]).dropna().to_numpy()
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append({"dataset": ds, "diff": d.mean(), "wins": int((d > 0).sum()),
                     "losses": int((d < 0).sum()), "n": len(d), "p": p})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


def endpoint_table(piv, specs) -> list[str]:
    head = "| Endpoint | " + " | ".join(NAMES[d] for d in DATASETS) + " |"
    lines = [head, "|---|" + "---|" * len(DATASETS)]
    detail = []
    for name, label, a, b in specs:
        t = compare(piv, a, b)
        cells = [f"{r['diff']:+.2f}{'*' if r.p_holm < 0.05 else ''}" for _, r in t.iterrows()]
        lines.append(f"| {name}: {label} | " + " | ".join(cells) + " |")
        detail.append(f"| {name} | " + " | ".join(
            f"{r.wins}–{r.losses}, p={r.p:.3f}, Holm {r.p_holm:.3f}" for _, r in t.iterrows()) + " |")
    lines += ["", "Wins–losses over the 10 seeds, raw p and Holm-adjusted p:", "",
              "| Endpoint | " + " | ".join(NAMES[d] for d in DATASETS) + " |",
              "|---|" + "---|" * len(DATASETS)] + detail
    return lines


def mean_table(runs, arms, value, fmt="{:.1f}") -> list[str]:
    piv = runs.pivot_table(index="arm", columns="dataset", values=value, aggfunc="mean")
    lines = ["| Arm | " + " | ".join(NAMES[d] for d in DATASETS) + " |",
             "|---|" + "---|" * len(DATASETS)]
    for a in arms:
        lines.append(f"| {SHORT[a]} | " + " | ".join(fmt.format(piv.loc[a, d]) for d in DATASETS) + " |")
    return lines


def labels_used(runs: pd.DataFrame) -> pd.DataFrame:
    """Labels per arm: pool labels bought, acceptance rows read by the gate, and the total.

    Computed from the split sizes, because the logged label columns are empty
    for Thyroid (it ran before they were added). Where they are logged, they
    must agree with this.
    """
    first = runs.groupby("dataset")[["n_pool", "n_accept", "budget"]].first()
    logged = runs.groupby(["dataset", "arm"])[["pool_labels", "gate_labels"]].mean()
    rows = []
    for ds in DATASETS:
        n_pool, n_acc, budget = first.loc[ds]
        gated = [a for a in runs.arm.unique() if a not in (B0, B9, SIMPSON)]
        for arm, pool, gate, check in ((B0, 0, 0, [B0]), (B9, n_pool, 0, [B9]), (SIMPSON, 0, 0, [SIMPSON]),
                                       ("gated arms (all OE and random arms)", budget, n_acc, gated)):
            for a in check:
                lp, lg = logged.loc[(ds, a)]
                assert np.isnan(lp) or (lp, lg) == (pool, gate), (ds, a, lp, lg)
            rows.append({"dataset": ds, "arm": arm, "pool": int(pool), "gate": int(gate),
                         "total": int(pool + gate)})
    return pd.DataFrame(rows)


def main() -> None:
    runs = pd.read_csv(RES / "runs.csv")
    rounds = pd.read_csv(RES / "rounds.csv")
    assert sorted(runs.seed.unique()) == list(range(10, 20)), "expected seeds 10-19"
    assert runs.groupby(["dataset", "arm"]).size().eq(10).all(), "expected 10 seeds per cell"
    piv = runs.pivot_table(index=["dataset", "seed"], columns="arm", values="test_macro_f1")

    out = ["# Refinement v1: summary of the pre-registered run", "",
           "Recomputed by `scripts/summarise_refinement_v1.py` from `runs.csv` and `rounds.csv` "
           "(copied unchanged from the teammate's run). Seeds 10–19, 6 datasets, 10 seeds each. "
           "Our own code reproduces the pre-registered arms exactly (Page Blocks, seed 10, all 7 arms).",
           "", "Metric: test macro-F1 (%) after the last round. Test: paired two-sided Wilcoxon over the "
           "10 seeds, Holm across the 6 datasets of each comparison. `*` = Holm-adjusted p < 0.05. "
           "Differences are in macro-F1 points, mean over seeds.", "",
           "## 1. Pre-registered endpoints", ""]
    out += endpoint_table(piv, ENDPOINTS)
    r1, r2 = compare(piv, *ENDPOINTS[0][2:]), compare(piv, *ENDPOINTS[1][2:])
    s1 = compare(piv, *ENDPOINTS[3][2:])
    out += ["", "**Reading:**", "",
            f"- **R1 (primary, RQ4):** the explanation's cut is not better than the smallest-cut rule on any "
            f"dataset ({int((r1.p_holm < 0.05).sum())} of 6 significant; mean difference "
            f"{r1['diff'].min():+.2f} to {r1['diff'].max():+.2f}). The blueprint hypothesis is not supported.",
            f"- **R2:** shrinking behind the gate beats add-only on {int(((r2.p_holm < 0.05) & (r2['diff'] > 0)).sum())} "
            f"of 6 datasets ({r2['diff'].min():+.2f} to {r2['diff'].max():+.2f} points).",
            f"- **S1:** order-ensemble selection beats random selection on "
            f"{int(((s1.p_holm < 0.05) & (s1['diff'] > 0)).sum())} of 6 datasets.",
            "", "## 2. Descriptive measures (pre-registered arms)", "",
            "### Test macro-F1 (%), mean over 10 seeds", ""]
    out += mean_table(runs, PREREG_ARMS, "test_macro_f1")
    out += ["", "### Lowest class recall (%), mean over 10 seeds", ""]
    out += mean_table(runs, PREREG_ARMS, "test_min_class_recall")
    out += ["", "### Test accuracy (%), mean over 10 seeds", ""]
    out += mean_table(runs, PREREG_ARMS, "test_accuracy")
    out += ["", "### Boxes at the end (mean)", ""]
    out += mean_table(runs, PREREG_ARMS, "boxes_end", "{:.0f}")

    gated = [a for a in PREREG_ARMS if a not in (B0, B9)]
    sub = runs[runs.arm.isin(gated)].copy()
    sub["hit_rate"] = 100 * sub.n_wrong_selected / sub.budget
    sub["shrink_accept"] = 100 * sub.n_shrink_accepted / sub.n_shrink_tried.replace(0, np.nan)
    for title, col, fmt in (("Share of picked points the model got wrong (%)", "hit_rate", "{:.1f}"),
                            ("Shrinks tried (mean per run)", "n_shrink_tried", "{:.0f}"),
                            ("Shrinks accepted by the gate (% of tried)", "shrink_accept", "{:.0f}"),
                            ("Mean share of the box's width cut by an accepted shrink", "mean_share_cut",
                             "{:.3f}"),
                            ("Mean training members cut per accepted shrink", "mean_rows_cut", "{:.0f}")):
        out += ["", f"### {title}", ""]
        out += mean_table(sub, [a for a in gated if a != OE + "A+B" or col == "hit_rate"], col, fmt)

    curve = rounds[rounds.arm.isin(PREREG_ARMS) & rounds["round"].isin([0, 5, 10])]
    cp = curve.pivot_table(index=["arm", "round"], columns="dataset", values="test_macro_f1")
    out += ["", "### Learning curve: test macro-F1 (%) at rounds 0, 5 and 10", "",
            "| Arm | Round | " + " | ".join(NAMES[d] for d in DATASETS) + " |",
            "|---|---|" + "---|" * len(DATASETS)]
    for a in gated:
        for r in (0, 5, 10):
            out.append(f"| {SHORT[a]} | {r} | " + " | ".join(f"{cp.loc[(a, r), d]:.1f}" for d in DATASETS) + " |")
    out += ["", "### Run time (seconds per run, mean)", ""]
    out += mean_table(runs, PREREG_ARMS, "time_s", "{:.0f}")

    lab = labels_used(runs)
    out += ["", "## 3. Every label each arm used", "",
            "The gate reads the labelled acceptance rows for every edit. They are labels too, and the "
            "pre-registered budget of 300 did not count them. B0 and B9 read none of them.", "",
            "| Dataset | Arm | Pool labels | Acceptance labels | Total |", "|---|---|---|---|---|"]
    for _, r in lab.iterrows():
        out.append(f"| {NAMES[r.dataset]} | {SHORT.get(r.arm, r.arm)} | {r.pool:,} | {r.gate:,} | {r.total:,} |")
    g = lab[lab.arm.str.startswith("gated")]
    ratio = g.gate / g.pool
    out += ["", f"The gated arms used {ratio.min():.1f}–{ratio.max():.0f}× as many acceptance labels as pool labels."]

    out += ["", "## Appendix: the two arms added by amendment (not pre-registered)", "",
            "Reported for completeness. Neither is used in the v2 proposal.", "",
            "- **C-minimal** keeps the features whose contrastive term favours the *wrong* class (the "
            "amendment describes it the other way round) and cuts the thinnest of them. It picks the "
            "same feature as C′-volume on 7–70% of eligible points, so R4 compares a half-geometric rule "
            "with the geometric rule. Its comparison with C′-volume was designed after the pilot had "
            "shown C′-volume winning, so it cannot replace R1 as the primary endpoint.",
            "- **B-Simpson** contracts overlapping boxes once after training. It finds no overlaps on "
            "Thyroid (zero-width sides on binary features are treated as open), so there it equals B0 "
            "exactly; elsewhere it drops thousands of training members and deletes boxes, without "
            "regrowing anything. It has no gate and uses no labels. It is not a fair stand-in for "
            "Simpson's contraction, so R6 and the Simpson rows say nothing about Simpson's method.", ""]
    out += endpoint_table(piv, AMENDED)
    out += ["", "### Test macro-F1 / lowest class recall (%), mean over 10 seeds", "",
            "| Arm | " + " | ".join(NAMES[d] for d in DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    f1 = runs.pivot_table(index="arm", columns="dataset", values="test_macro_f1")
    mr = runs.pivot_table(index="arm", columns="dataset", values="test_min_class_recall")
    for a in (SIMPSON, CMIN):
        out.append(f"| {SHORT[a]} | " + " | ".join(f"{f1.loc[a, d]:.1f} / {mr.loc[a, d]:.1f}"
                                                  for d in DATASETS) + " |")
    (RES / "summary_v1.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {RES / 'summary_v1.md'}")


if __name__ == "__main__":
    main()
